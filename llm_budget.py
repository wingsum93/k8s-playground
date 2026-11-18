#!/usr/bin/env python3
"""Check OpenRouter credit usage and DeepSeek balances and billing exports."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, DecimalException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


UTC = timezone.utc
OPENROUTER_API = "https://openrouter.ai/api/v1"
DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"
OPENROUTER_COUNT_METRICS = (
    "request_count", "cached_tokens", "tokens_prompt", "tokens_completion", "tokens_total"
)
DEEPSEEK_COUNT_TYPES = (
    "request_count", "input_cache_hit_tokens", "input_cache_miss_tokens", "output_tokens"
)
COUNT_HEADERS = (
    "Requests", "Cached Input Tokens", "Input Tokens", "Output Tokens", "Total Tokens"
)
PROVIDERS = ("openrouter", "deepseek", "both")


def request_json(url, key, payload=None):
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    try:
        with urlopen(Request(url, data=data, headers=headers), timeout=20) as response:
            return json.load(response, parse_float=Decimal)
    except HTTPError as error:
        if error.code in (401, 403) and url.startswith(OPENROUTER_API):
            raise ValueError(
                f"HTTP {error.code}. Set OPENROUTER_API_KEY to a valid Management Key."
            ) from None
        raise ValueError(f"HTTP {error.code}.") from None
    except (URLError, TimeoutError):
        raise ValueError("Could not reach the provider or the request timed out.") from None


def reporting_windows(now, report_timezone):
    local_month_start = now.astimezone(report_timezone).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    month_start = local_month_start.astimezone(UTC)
    last_month_start = (local_month_start - timedelta(days=1)).replace(day=1)
    return (
        ("Past 1 hour", now - timedelta(hours=1), now),
        ("Past 3 hours", now - timedelta(hours=3), now),
        ("Past 1 day", now - timedelta(days=1), now),
        ("Past 1 week", now - timedelta(weeks=1), now),
        ("This month", month_start, now),
        ("Last month", last_month_start.astimezone(UTC), month_start),
    )


def utc_timestamp(value):
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def openrouter_usage(key, start, end):
    if start == end:
        return Decimal(0), (0,) * len(OPENROUTER_COUNT_METRICS)

    payload = {
        "metrics": ["credits_usage", *OPENROUTER_COUNT_METRICS],
        "time_range": {"start": utc_timestamp(start), "end": utc_timestamp(end)},
    }
    duration = end - start
    if duration <= timedelta(hours=3):
        payload["granularity"] = "minute"
    elif duration <= timedelta(days=31):
        payload["granularity"] = "hour"

    result = request_json(f"{OPENROUTER_API}/analytics/query", key, payload)["data"]
    if result["metadata"]["truncated"] or result.get("warnings"):
        raise ValueError("Analytics returned incomplete results or query warnings.")

    rows = result["data"]
    return (
        sum((Decimal(row["credits_usage"]) for row in rows), Decimal(0)),
        tuple(sum(int(row[metric]) for row in rows) for metric in OPENROUTER_COUNT_METRICS),
    )


def print_usage_table(spend_header, rows):
    table = [("Window", spend_header, *COUNT_HEADERS, "")]
    for label, spend, counts, note in rows:
        formatted_counts = [f"{count:,}" if count is not None else "N/A" for count in counts]
        table.append((label, spend, *formatted_counts, note))

    widths = [max(map(len, column)) for column in zip(*table)]
    for label, *values, note in table:
        columns = [label.ljust(widths[0])]
        columns.extend(value.rjust(width) for value, width in zip(values, widths[1:-1]))
        print("  ".join([*columns, note]).rstrip())


def show_openrouter(key, windows):
    print("OpenRouter")
    rows = []
    try:
        for label, start, end in windows:
            cost, counts = openrouter_usage(key, start, end)
            rows.append((label, f"{cost:,.6f}", counts, ""))
    finally:
        if rows:
            print_usage_table("Credits spent (USD)", rows)

    credits = request_json(f"{OPENROUTER_API}/credits", key)["data"]
    remaining = Decimal(credits["total_credits"]) - Decimal(credits["total_usage"])
    print(f"Remaining credits: {remaining:,.6f} USD")


def parse_timestamp(value):
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Timestamp requires a UTC offset.")
    return timestamp


def read_deepseek_csv(path, value_columns, parse_value):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        columns = set(reader.fieldnames or ())
        has_intervals = {"start_time_iso", "end_time_iso"} <= columns
        has_dates = has_intervals or "utc_date" in columns
        if not (value_columns <= columns and has_dates):
            required = ", ".join(sorted(value_columns))
            raise ValueError(f"{path.name} must be a DeepSeek CSV with dates and {required}.")

        for row in reader:
            try:
                if has_intervals:
                    start = parse_timestamp(row["start_time_iso"].strip())
                    end = parse_timestamp(row["end_time_iso"].strip())
                else:
                    start = datetime.strptime(row["utc_date"].strip(), "%Y-%m-%d").replace(
                        tzinfo=UTC
                    )
                    end = start + timedelta(days=1)
                if start >= end:
                    raise ValueError("Invalid billing interval.")
                value = parse_value(row)
            except (AttributeError, ValueError, DecimalException):
                raise ValueError(
                    f"{path.name} has invalid billing data on line {reader.line_num}."
                ) from None
            if value is not None:
                yield start, end, *value


def parse_deepseek_cost(row):
    currency = row["currency"].strip().upper()
    cost = Decimal(row["cost"].strip())
    if not currency or not cost.is_finite():
        raise ValueError("Invalid billing cost.")
    return currency, cost


def parse_deepseek_amount(row):
    amount_type = row["type"].strip()
    if amount_type in DEEPSEEK_COUNT_TYPES:
        return amount_type, int(row["amount"])
    return None


def deepseek_totals(rows, start, end, now):
    totals = defaultdict(int)
    partial = False
    if start == end:
        return totals, partial
    for bucket_start, bucket_end, category, value in rows:
        if bucket_end <= start or bucket_start >= end:
            continue
        # A current bucket only contains usage recorded when the CSV was exported.
        if bucket_start < start or (end < now and bucket_end > end):
            partial = True
            continue
        totals[category] += value
    return totals, partial


def deepseek_usage_rows(cost_rows, amount_rows, windows, now):
    for label, start, end in windows:
        costs, partial_cost = deepseek_totals(cost_rows, start, end, now)
        amounts, partial_amount = deepseek_totals(amount_rows, start, end, now)
        spend = ", ".join(f"{cost:,.6f} {currency}" for currency, cost in sorted(costs.items()))
        counts = (None,) * len(COUNT_HEADERS)
        if amounts:
            requests, cached, uncached, output = (amounts[kind] for kind in DEEPSEEK_COUNT_TYPES)
            input_tokens = cached + uncached
            counts = (requests, cached, input_tokens, output, input_tokens + output)

        excluded = []
        if partial_cost:
            excluded.append("spend")
        if partial_amount:
            excluded.append("counts")
        note = f"Partial {' and '.join(excluded)} (boundary buckets omitted)" if excluded else ""
        yield label, spend or "N/A", counts, note


def show_deepseek(key, cost_csv, amount_csv, windows, now):
    print("DeepSeek")
    if not cost_csv:
        print("Supply a cost export with --deepseek-cost-csv for historical spend.")
    if not amount_csv:
        print("Supply an amount export with --deepseek-amount-csv for historical token counts.")
    if cost_csv or amount_csv:
        print("Historical values reflect the supplied CSV exports.")

    costs = [
        row for path in cost_csv
        for row in read_deepseek_csv(path, {"cost", "currency"}, parse_deepseek_cost)
    ]
    amounts = [
        row for path in amount_csv
        for row in read_deepseek_csv(path, {"type", "amount"}, parse_deepseek_amount)
    ]
    print_usage_table("Spend", deepseek_usage_rows(costs, amounts, windows, now))

    if key:
        balance = request_json(DEEPSEEK_BALANCE_URL, key)
        for entry in balance["balance_infos"]:
            print(f"Current balance: {Decimal(entry['total_balance']):,.6f} {entry['currency']}")
        if not balance["is_available"]:
            print("DeepSeek reports insufficient balance for API calls.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=PROVIDERS, nargs="?",
        help="Choose one provider or run both (default: run configured providers).",
    )
    parser.add_argument("--provider", choices=PROVIDERS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--deepseek-cost-csv", type=Path, action="append", default=[],
        help="DeepSeek billing cost CSV. Repeat for exports with non-overlapping dates.",
    )
    parser.add_argument(
        "--deepseek-amount-csv", type=Path, action="append", default=[],
        help="DeepSeek usage amount CSV. Repeat for exports with non-overlapping dates.",
    )
    parser.add_argument(
        "--timezone", default="Asia/Hong_Kong",
        help="IANA time zone for the calendar month (default: Asia/Hong_Kong).",
    )
    args = parser.parse_args()
    if args.command and args.provider and args.command != args.provider:
        parser.error("The command and --provider must select the same provider.")
    provider = args.command or args.provider
    try:
        report_timezone = ZoneInfo(args.timezone)
    except (ValueError, ZoneInfoNotFoundError):
        parser.error(f"Unknown time zone: {args.timezone}")

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    has_deepseek_input = bool(deepseek_key or args.deepseek_cost_csv or args.deepseek_amount_csv)
    if provider in ("openrouter", "both") and not openrouter_key:
        parser.error("Set OPENROUTER_API_KEY to an OpenRouter Management Key.")
    if provider in ("deepseek", "both") and not has_deepseek_input:
        parser.error("Set DEEPSEEK_API_KEY or provide a DeepSeek cost or amount CSV.")
    if not (openrouter_key or has_deepseek_input):
        parser.error("Set OPENROUTER_API_KEY or DEEPSEEK_API_KEY, or provide a DeepSeek CSV.")

    now = datetime.now(UTC).replace(microsecond=0)
    windows = reporting_windows(now, report_timezone)
    print(f"Report time: {now.astimezone(report_timezone).isoformat()} ({args.timezone})")
    print("Input token counts include cached input tokens.")

    providers = []
    if openrouter_key and provider != "deepseek":
        providers.append(("OpenRouter", show_openrouter, (openrouter_key, windows)))
    if has_deepseek_input and provider != "openrouter":
        inputs = (deepseek_key, args.deepseek_cost_csv, args.deepseek_amount_csv, windows, now)
        providers.append(("DeepSeek", show_deepseek, inputs))

    exit_code = 0
    for name, show, parameters in providers:
        print()
        try:
            show(*parameters)
        except (OSError, ValueError, DecimalException, csv.Error) as error:
            print(f"{name}: {error}", file=sys.stderr)
            exit_code = 1
        except (KeyError, TypeError):
            print(f"{name}: Unexpected response format.", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
