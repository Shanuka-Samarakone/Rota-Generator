from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend import MINIMUM_REST_HOURS, SHIFT_DETAILS
from data_handler import load_excel_data


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    message: str


def validate_rota(*, rota_file: str | Path, input_file: str | Path) -> list[ValidationCheck]:
    """Validate a generated rota against the original input workbook."""

    rota_path = Path(rota_file).expanduser().resolve()
    input_path = Path(input_file).expanduser().resolve()

    if not rota_path.exists():
        raise FileNotFoundError(f"The generated rota file could not be found:\n{rota_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"The input workbook could not be found:\n{input_path}")

    rota = pd.read_excel(rota_path, sheet_name="Generated_Rota", engine="openpyxl")
    data = load_excel_data(input_path)
    rota = prepare_rota(rota)

    return [
        check_required_columns(rota),
        check_duplicate_employee_dates(rota),
        check_one_shift_per_day(rota),
        check_approved_leave(rota, data["leave"]),
        check_unavailability(rota, data["unavailability"]),
        check_minimum_staffing(rota, data["store_requirements"]),
        check_manager_coverage(rota, data["employees"], data["store_requirements"]),
        check_till_coverage(rota, data["employees"], data["store_requirements"]),
        check_opening_coverage(rota, data["employees"], data["store_requirements"]),
        check_closing_coverage(rota, data["employees"], data["store_requirements"]),
        check_rest_periods(rota),
        check_seven_day_limit(rota),
        check_weekly_hours(rota, data["employees"], data["leave"]),
    ]


def prepare_rota(rota: pd.DataFrame) -> pd.DataFrame:
    rota = rota.copy()
    rota.columns = (
        rota.columns.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    rota["date"] = pd.to_datetime(rota["date"], errors="raise").dt.normalize()
    rota["employee_id"] = rota["employee_id"].astype(str).str.strip()
    rota["shift"] = rota["shift"].astype(str).str.strip().str.upper()
    rota["hours"] = pd.to_numeric(rota["hours"], errors="raise")
    return rota


def check_required_columns(rota: pd.DataFrame) -> ValidationCheck:
    required = {"employee_id", "employee_name", "role", "date", "shift", "hours"}
    missing = sorted(required - set(rota.columns))
    if missing:
        return ValidationCheck("Required output columns", False, "Missing columns: " + ", ".join(missing))
    return ValidationCheck("Required output columns", True, "All required columns are present.")


def check_duplicate_employee_dates(rota: pd.DataFrame) -> ValidationCheck:
    duplicates = rota.duplicated(subset=["employee_id", "date"], keep=False)
    if duplicates.any():
        return ValidationCheck(
            "One record per employee per date",
            False,
            f"{int(duplicates.sum())} duplicate employee-date rows were found.",
        )
    return ValidationCheck("One record per employee per date", True, "No duplicate employee-date rows were found.")


def check_one_shift_per_day(rota: pd.DataFrame) -> ValidationCheck:
    counts = rota[rota["shift"] != "OFF"].groupby(["employee_id", "date"]).size()
    violations = counts[counts > 1]
    if not violations.empty:
        return ValidationCheck(
            "Maximum one shift per employee per day",
            False,
            f"{len(violations)} employee-date combinations contain more than one shift.",
        )
    return ValidationCheck("Maximum one shift per employee per day", True, "No employee has more than one shift on a date.")


def check_approved_leave(rota: pd.DataFrame, leave: pd.DataFrame) -> ValidationCheck:
    approved = leave[leave["status"].astype(str).str.upper() == "APPROVED"]
    violations = 0
    for row in approved.itertuples(index=False):
        employee = str(row.employee_id)
        matches = rota[
            (rota["employee_id"] == employee)
            & (rota["date"] >= pd.Timestamp(row.start_date))
            & (rota["date"] <= pd.Timestamp(row.end_date))
            & (rota["shift"] != "OFF")
        ]
        violations += len(matches)
    if violations:
        return ValidationCheck("Approved leave", False, f"{violations} working assignments were found during approved leave.")
    return ValidationCheck("Approved leave", True, "No employee is scheduled during approved leave.")


def check_unavailability(rota: pd.DataFrame, unavailability: pd.DataFrame) -> ValidationCheck:
    lookup = (
        unavailability.assign(employee_id=lambda frame: frame["employee_id"].astype(str))
        .set_index("employee_id")
        .to_dict("index")
    )
    violations = 0
    for row in rota[rota["shift"] != "OFF"].itertuples(index=False):
        weekday = row.date.day_name().lower()
        if bool(lookup.get(str(row.employee_id), {}).get(weekday, False)):
            violations += 1
    if violations:
        return ValidationCheck("Recurring unavailability", False, f"{violations} assignments were made on unavailable weekdays.")
    return ValidationCheck("Recurring unavailability", True, "Recurring unavailable weekdays were respected.")


def check_minimum_staffing(rota: pd.DataFrame, requirements: dict[str, int]) -> ValidationCheck:
    minimum_staff = int(requirements["minimum_staff"])
    daily_counts = rota[rota["shift"] != "OFF"].groupby("date").size()
    violations = daily_counts[daily_counts < minimum_staff]
    if not violations.empty:
        return ValidationCheck(
            "Minimum daily staffing",
            False,
            f"{len(violations)} dates have fewer than {minimum_staff} working employees.",
        )
    return ValidationCheck("Minimum daily staffing", True, f"Every date has at least {minimum_staff} employees.")


def check_manager_coverage(rota: pd.DataFrame, employees: pd.DataFrame, requirements: dict[str, int]) -> ValidationCheck:
    manager_roles = {"store manager", "deputy manager", "department manager", "manager"}
    managers = set(
        employees.loc[
            employees["role"].astype(str).str.strip().str.lower().isin(manager_roles),
            "employee_id",
        ].astype(str)
    )
    return check_daily_group_coverage(
        rota=rota,
        eligible_employees=managers,
        required=int(requirements["minimum_managers"]),
        name="Manager coverage",
    )


def check_till_coverage(rota: pd.DataFrame, employees: pd.DataFrame, requirements: dict[str, int]) -> ValidationCheck:
    till_staff = set(employees.loc[employees["till_trained"].astype(bool), "employee_id"].astype(str))
    return check_daily_group_coverage(
        rota=rota,
        eligible_employees=till_staff,
        required=int(requirements["minimum_till_staff"]),
        name="Till-trained coverage",
    )


def check_opening_coverage(rota: pd.DataFrame, employees: pd.DataFrame, requirements: dict[str, int]) -> ValidationCheck:
    openers = set(employees.loc[employees["can_open"].astype(bool), "employee_id"].astype(str))
    return check_daily_group_coverage(
        rota=rota[rota["shift"].isin(["OPEN", "SHORT_OPEN"])],
        eligible_employees=openers,
        required=int(requirements["minimum_openers"]),
        name="Opening coverage",
    )


def check_closing_coverage(rota: pd.DataFrame, employees: pd.DataFrame, requirements: dict[str, int]) -> ValidationCheck:
    closers = set(employees.loc[employees["can_close"].astype(bool), "employee_id"].astype(str))
    return check_daily_group_coverage(
        rota=rota[rota["shift"].isin(["CLOSE", "SHORT_CLOSE"])],
        eligible_employees=closers,
        required=int(requirements["minimum_closers"]),
        name="Closing coverage",
    )


def check_daily_group_coverage(*, rota: pd.DataFrame, eligible_employees: set[str], required: int, name: str) -> ValidationCheck:
    all_dates = sorted(rota["date"].dropna().unique())
    eligible_working = rota[
        (rota["shift"] != "OFF")
        & rota["employee_id"].astype(str).isin(eligible_employees)
    ]
    counts = eligible_working.groupby("date").size().reindex(all_dates, fill_value=0)
    violations = counts[counts < required]
    if not violations.empty:
        return ValidationCheck(name, False, f"{len(violations)} dates have fewer than {required} eligible employees.")
    return ValidationCheck(name, True, f"Every date has at least {required} eligible employees.")


def check_rest_periods(rota: pd.DataFrame) -> ValidationCheck:
    working = rota[rota["shift"] != "OFF"].sort_values(["employee_id", "date"])
    violations = 0
    for _, employee_rota in working.groupby("employee_id"):
        rows = list(employee_rota.itertuples(index=False))
        for current, following in zip(rows, rows[1:]):
            if (following.date - current.date).days != 1:
                continue
            first = SHIFT_DETAILS.get(current.shift)
            second = SHIFT_DETAILS.get(following.shift)
            if first is None or second is None:
                continue
            rest_hours = 24 - first["end"] + second["start"]
            if rest_hours < MINIMUM_REST_HOURS:
                violations += 1
    if violations:
        return ValidationCheck(
            "Minimum rest between consecutive shifts",
            False,
            f"{violations} consecutive-shift rest violations were found.",
        )
    return ValidationCheck(
        "Minimum rest between consecutive shifts",
        True,
        f"All consecutive shifts provide at least {MINIMUM_REST_HOURS} hours of rest.",
    )


def check_seven_day_limit(rota: pd.DataFrame) -> ValidationCheck:
    violations = set()
    for employee, employee_rota in rota.groupby("employee_id"):
        working = (
            employee_rota.sort_values("date").set_index("date")["shift"] != "OFF"
        ).astype(int)
        if len(working) >= 7 and (working.rolling(window=7).sum() > 6).any():
            violations.add(employee)
    if violations:
        return ValidationCheck(
            "Maximum six working days in seven",
            False,
            f"{len(violations)} employees worked all seven days in at least one rolling seven-day period.",
        )
    return ValidationCheck(
        "Maximum six working days in seven",
        True,
        "No employee works all seven days in a rolling seven-day period.",
    )


def check_weekly_hours(rota: pd.DataFrame, employees: pd.DataFrame, leave: pd.DataFrame) -> ValidationCheck:
    contract_lookup = (
        employees.assign(employee_id=lambda frame: frame["employee_id"].astype(str))
        .set_index("employee_id")["contract_hours"]
        .astype(int)
        .to_dict()
    )
    approved_lookup = create_approved_leave_lookup(list(contract_lookup), leave)
    rota = rota.copy()
    iso = rota["date"].dt.isocalendar()
    rota["iso_year"] = iso.year.astype(int)
    rota["iso_week"] = iso.week.astype(int)
    violations = 0

    for (employee, _, _), week_data in rota.groupby(["employee_id", "iso_year", "iso_week"]):
        dates = list(week_data["date"].drop_duplicates())
        approved_days = sum(day in approved_lookup.get(str(employee), set()) for day in dates)
        available_days = max(len(dates) - approved_days, 0)
        expected = contract_lookup[str(employee)] * available_days / 7
        expected_hours = int(round(expected / 4) * 4)
        actual_hours = int(week_data["hours"].sum())
        if actual_hours != expected_hours:
            violations += 1

    if violations:
        return ValidationCheck(
            "Weekly contracted hours",
            False,
            f"{violations} employee-week hour differences were found.",
        )
    return ValidationCheck(
        "Weekly contracted hours",
        True,
        "All employee-week hours match the backend's current target-hours calculation.",
    )


def create_approved_leave_lookup(employees: list[str], leave: pd.DataFrame) -> dict[str, set[pd.Timestamp]]:
    lookup = {str(employee): set() for employee in employees}
    approved = leave[leave["status"].astype(str).str.upper() == "APPROVED"]
    for row in approved.itertuples(index=False):
        employee = str(row.employee_id)
        if employee in lookup:
            lookup[employee].update(
                pd.Timestamp(day).normalize()
                for day in pd.date_range(row.start_date, row.end_date, freq="D")
            )
    return lookup


def print_validation_report(checks: list[ValidationCheck]) -> None:
    print()
    print("=" * 70)
    print("ROTA VALIDATION REPORT")
    print("=" * 70)

    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status:<4} | {check.name}")
        print(f"       {check.message}")
        print("-" * 70)

    passed_count = sum(check.passed for check in checks)
    failed_count = len(checks) - passed_count

    print()
    print(f"Checks passed: {passed_count}/{len(checks)}")
    print(f"Checks failed: {failed_count}")
    print("Overall result:", "PASS" if failed_count == 0 else "FAIL")


if __name__ == "__main__":
    results = validate_rota(
        rota_file="output/generated_rota_2027.xlsx",
        input_file="DATA AND LEAVE.xlsx",
    )
    print_validation_report(results)