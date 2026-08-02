from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from ortools.sat.python import cp_model

from data_handler import load_excel_data
from output_handler import save_rota_to_excel


# ==========================================================
# CONFIGURATION
# ==========================================================

SHIFT_DETAILS = {
    "OPEN": {"start": 9, "end": 17, "hours": 8},
    "MID": {"start": 12, "end": 20, "hours": 8},
    "CLOSE": {"start": 14, "end": 22, "hours": 8},
    "SHORT_OPEN": {"start": 9, "end": 13, "hours": 4},
    "SHORT_CLOSE": {"start": 18, "end": 22, "hours": 4},
}

SHIFTS = list(SHIFT_DETAILS)

MANAGER_ROLES = {
    "store manager",
    "deputy manager",
    "department manager",
    "manager",
}

MINIMUM_REST_HOURS = 12

REQUESTED_LEAVE_WEIGHT = 100
CLOSING_FAIRNESS_WEIGHT = 10
WEEKEND_FAIRNESS_WEIGHT = 5


ProgressCallback = Callable[[str], None]


# ==========================================================
# RESULT RETURNED TO THE FRONTEND
# ==========================================================

@dataclass
class RotaResult:
    success: bool
    status: str
    message: str
    rota: pd.DataFrame | None = None
    output_file: Path | None = None
    objective_value: float | None = None
    solve_time_seconds: float = 0.0


# ==========================================================
# PUBLIC FUNCTION
# ==========================================================

def generate_rota(
    *,
    start_date: str,
    end_date: str,
    input_file: str | Path | None = None,
    data: dict[str, Any] | None = None,
    output_file: str | Path | None = None,
    maximum_solve_time: int = 120,
    progress_callback: ProgressCallback | None = None,
) -> RotaResult:
    """
    Generate a rota for every date from start_date to end_date.

    The frontend will eventually call this function.

    Data can currently come from either:
    1. input_file: the existing Excel workbook; or
    2. data: a dictionary supplied by a future database/frontend.

    Exactly one of input_file or data must be provided.
    """

    report_progress(progress_callback, "Validating rota settings...")

    schedule_dates = create_schedule_dates(
        start_date=start_date,
        end_date=end_date,
    )

    if (input_file is None) == (data is None):
        raise ValueError(
            "Provide exactly one data source: either input_file or data."
        )

    report_progress(progress_callback, "Loading employee data...")

    rota_data = (
        load_excel_data(input_file)
        if input_file is not None
        else validate_supplied_data(data)
    )

    employees_df = rota_data["employees"].copy()
    leave_df = rota_data["leave"].copy()
    unavailability_df = rota_data["unavailability"].copy()
    store_requirements = dict(
        rota_data["store_requirements"]
    )

    employees = (
        employees_df["employee_id"]
        .astype(str)
        .tolist()
    )

    validate_model_inputs(
        employees_df=employees_df,
        leave_df=leave_df,
        unavailability_df=unavailability_df,
        store_requirements=store_requirements,
    )

    report_progress(
        progress_callback,
        f"Building a rota for {len(employees)} employees "
        f"across {len(schedule_dates)} days...",
    )

    model = cp_model.CpModel()

    work = create_decision_variables(
        model=model,
        employees=employees,
        schedule_dates=schedule_dates,
    )

    add_one_shift_per_day_constraints(
        model=model,
        work=work,
        employees=employees,
        schedule_dates=schedule_dates,
    )

    add_approved_leave_constraints(
        model=model,
        work=work,
        schedule_dates=schedule_dates,
        leave_df=leave_df,
    )

    add_unavailability_constraints(
        model=model,
        work=work,
        employees=employees,
        schedule_dates=schedule_dates,
        unavailability_df=unavailability_df,
    )

    add_weekly_hours_constraints(
        model=model,
        work=work,
        employees=employees,
        schedule_dates=schedule_dates,
        employees_df=employees_df,
        leave_df=leave_df,
    )

    add_rest_constraints(
        model=model,
        work=work,
        employees=employees,
        schedule_dates=schedule_dates,
    )

    add_seven_day_working_limit(
        model=model,
        work=work,
        employees=employees,
        schedule_dates=schedule_dates,
    )

    add_operational_constraints(
        model=model,
        work=work,
        employees=employees,
        schedule_dates=schedule_dates,
        employees_df=employees_df,
        store_requirements=store_requirements,
    )

    penalty_terms = add_soft_constraints(
        model=model,
        work=work,
        employees=employees,
        schedule_dates=schedule_dates,
        employees_df=employees_df,
        leave_df=leave_df,
    )

    if penalty_terms:
        model.Minimize(sum(penalty_terms))

    report_progress(progress_callback, "Running the rota optimiser...")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = maximum_solve_time
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return failed_result(
            status=status,
            status_name=status_name,
            solver=solver,
        )

    report_progress(progress_callback, "Creating the rota table...")

    rota = extract_rota(
        solver=solver,
        work=work,
        employees=employees,
        schedule_dates=schedule_dates,
        employees_df=employees_df,
    )

    saved_path = None

    if output_file is not None:
        report_progress(progress_callback, "Saving the rota to Excel...")
        saved_path = save_rota_to_excel(
            rota=rota,
            output_path=output_file,
        )

    report_progress(progress_callback, "Rota generation completed.")

    message = (
        "An optimal rota was generated."
        if status == cp_model.OPTIMAL
        else "A valid rota was generated within the allowed solving time."
    )

    return RotaResult(
        success=True,
        status=status_name,
        message=message,
        rota=rota,
        output_file=saved_path,
        objective_value=solver.ObjectiveValue(),
        solve_time_seconds=solver.WallTime(),
    )


# ==========================================================
# DATES AND DATA VALIDATION
# ==========================================================

def create_schedule_dates(
    start_date: str,
    end_date: str,
) -> list[pd.Timestamp]:
    """Create every calendar date in the requested period."""

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if end < start:
        raise ValueError(
            "The rota end date cannot be earlier than the start date."
        )

    return list(pd.date_range(start=start, end=end, freq="D"))


def validate_supplied_data(
    data: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Validate data supplied directly by a future frontend/database.
    """

    if data is None:
        raise ValueError("No rota data was supplied.")

    required_keys = {
        "employees",
        "department_skills",
        "leave",
        "unavailability",
        "store_requirements",
    }

    missing = sorted(required_keys - set(data))

    if missing:
        raise ValueError(
            "The supplied rota data is missing: "
            + ", ".join(missing)
        )

    return data


def validate_model_inputs(
    employees_df: pd.DataFrame,
    leave_df: pd.DataFrame,
    unavailability_df: pd.DataFrame,
    store_requirements: dict[str, int],
) -> None:
    """Run final checks before building the optimisation model."""

    if employees_df.empty:
        raise ValueError("No employees were found.")

    if employees_df["employee_id"].duplicated().any():
        raise ValueError("Employee IDs must be unique.")

    employee_ids = set(
        employees_df["employee_id"].astype(str)
    )

    for sheet_name, dataframe in {
        "Leave": leave_df,
        "Unavailability": unavailability_df,
    }.items():
        referenced = set(
            dataframe["employee_id"]
            .dropna()
            .astype(str)
        )

        unknown = sorted(referenced - employee_ids)

        if unknown:
            raise ValueError(
                f"{sheet_name} contains unknown employee IDs: "
                + ", ".join(unknown)
            )

    required_requirements = {
        "minimum_staff",
        "minimum_managers",
        "minimum_till_staff",
        "minimum_openers",
        "minimum_closers",
    }

    missing_requirements = sorted(
        required_requirements - set(store_requirements)
    )

    if missing_requirements:
        raise ValueError(
            "Store requirements are missing: "
            + ", ".join(missing_requirements)
        )


# ==========================================================
# DECISION VARIABLES
# ==========================================================

def create_decision_variables(
    model: cp_model.CpModel,
    employees: list[str],
    schedule_dates: list[pd.Timestamp],
) -> dict[tuple[str, pd.Timestamp, str], cp_model.IntVar]:
    """
    work[(employee, date, shift)] equals:
    1 when assigned and 0 when not assigned.
    """

    return {
        (employee, day, shift): model.NewBoolVar(
            f"work_{employee}_{day:%Y%m%d}_{shift}"
        )
        for employee in employees
        for day in schedule_dates
        for shift in SHIFTS
    }


# ==========================================================
# HARD CONSTRAINTS
# ==========================================================

def add_one_shift_per_day_constraints(
    model: cp_model.CpModel,
    work: dict,
    employees: list[str],
    schedule_dates: list[pd.Timestamp],
) -> None:
    """An employee may work at most one shift per day."""

    for employee in employees:
        for day in schedule_dates:
            model.Add(
                sum(
                    work[(employee, day, shift)]
                    for shift in SHIFTS
                )
                <= 1
            )


def add_approved_leave_constraints(
    model: cp_model.CpModel,
    work: dict,
    schedule_dates: list[pd.Timestamp],
    leave_df: pd.DataFrame,
) -> None:
    """Approved leave cannot be overridden by the solver."""

    schedule_set = set(schedule_dates)

    approved = leave_df[
        leave_df["status"].astype(str).str.upper()
        == "APPROVED"
    ]

    for row in approved.itertuples(index=False):
        employee = str(row.employee_id)

        for day in pd.date_range(
            row.start_date,
            row.end_date,
            freq="D",
        ):
            day = pd.Timestamp(day).normalize()

            if day not in schedule_set:
                continue

            model.Add(
                sum(
                    work[(employee, day, shift)]
                    for shift in SHIFTS
                )
                == 0
            )


def add_unavailability_constraints(
    model: cp_model.CpModel,
    work: dict,
    employees: list[str],
    schedule_dates: list[pd.Timestamp],
    unavailability_df: pd.DataFrame,
) -> None:
    """Respect recurring unavailable weekdays."""

    lookup = (
        unavailability_df
        .assign(
            employee_id=lambda frame:
            frame["employee_id"].astype(str)
        )
        .set_index("employee_id")
        .to_dict("index")
    )

    for employee in employees:
        employee_rules = lookup.get(employee, {})

        for day in schedule_dates:
            weekday = day.day_name().lower()

            if bool(employee_rules.get(weekday, False)):
                model.Add(
                    sum(
                        work[(employee, day, shift)]
                        for shift in SHIFTS
                    )
                    == 0
                )


def add_weekly_hours_constraints(
    model: cp_model.CpModel,
    work: dict,
    employees: list[str],
    schedule_dates: list[pd.Timestamp],
    employees_df: pd.DataFrame,
    leave_df: pd.DataFrame,
) -> None:
    """
    Match contracted hours within each calendar week.

    For partial weeks at the beginning or end, hours are reduced
    proportionally. Approved leave also reduces the required worked
    hours for that week.
    """

    contract_hours = (
        employees_df
        .assign(
            employee_id=lambda frame:
            frame["employee_id"].astype(str)
        )
        .set_index("employee_id")["contract_hours"]
        .astype(int)
        .to_dict()
    )

    approved_leave = create_approved_leave_lookup(
        employees=employees,
        leave_df=leave_df,
    )

    for week_dates in group_dates_by_week(
        schedule_dates
    ).values():
        calendar_days_in_model = len(week_dates)

        for employee in employees:
            approved_days = sum(
                day in approved_leave[employee]
                for day in week_dates
            )

            available_model_days = max(
                calendar_days_in_model - approved_days,
                0,
            )

            target = (
                contract_hours[employee]
                * available_model_days
                / 7
            )

            target_hours = int(round(target / 4) * 4)

            maximum_possible = (
                available_model_days
                * max(
                    details["hours"]
                    for details in SHIFT_DETAILS.values()
                )
            )

            target_hours = min(
                target_hours,
                maximum_possible,
            )

            weekly_hours = sum(
                work[(employee, day, shift)]
                * SHIFT_DETAILS[shift]["hours"]
                for day in week_dates
                for shift in SHIFTS
            )

            model.Add(weekly_hours == target_hours)


def add_rest_constraints(
    model: cp_model.CpModel,
    work: dict,
    employees: list[str],
    schedule_dates: list[pd.Timestamp],
) -> None:
    """Prevent consecutive shifts with insufficient rest."""

    incompatible_pairs = []

    for first_shift in SHIFTS:
        for second_shift in SHIFTS:
            rest = (
                24
                - SHIFT_DETAILS[first_shift]["end"]
                + SHIFT_DETAILS[second_shift]["start"]
            )

            if rest < MINIMUM_REST_HOURS:
                incompatible_pairs.append(
                    (first_shift, second_shift)
                )

    dates = sorted(schedule_dates)

    for employee in employees:
        for index in range(len(dates) - 1):
            today = dates[index]
            tomorrow = dates[index + 1]

            for first_shift, second_shift in incompatible_pairs:
                model.Add(
                    work[(employee, today, first_shift)]
                    + work[(employee, tomorrow, second_shift)]
                    <= 1
                )


def add_seven_day_working_limit(
    model: cp_model.CpModel,
    work: dict,
    employees: list[str],
    schedule_dates: list[pd.Timestamp],
) -> None:
    """
    Prevent an employee from working all seven days in any
    rolling seven-day period.
    """

    dates = sorted(schedule_dates)

    if len(dates) < 7:
        return

    for employee in employees:
        for start_index in range(len(dates) - 6):
            seven_days = dates[
                start_index:start_index + 7
            ]

            model.Add(
                sum(
                    work[(employee, day, shift)]
                    for day in seven_days
                    for shift in SHIFTS
                )
                <= 6
            )


def add_operational_constraints(
    model: cp_model.CpModel,
    work: dict,
    employees: list[str],
    schedule_dates: list[pd.Timestamp],
    employees_df: pd.DataFrame,
    store_requirements: dict[str, int],
) -> None:
    """Ensure each day has the required staff and skills."""

    employee_table = employees_df.copy()
    employee_table["employee_id"] = (
        employee_table["employee_id"].astype(str)
    )
    employee_table["normalised_role"] = (
        employee_table["role"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    managers = employee_table.loc[
        employee_table["normalised_role"].isin(
            MANAGER_ROLES
        ),
        "employee_id",
    ].tolist()

    till_staff = employee_table.loc[
        employee_table["till_trained"].astype(bool),
        "employee_id",
    ].tolist()

    openers = employee_table.loc[
        employee_table["can_open"].astype(bool),
        "employee_id",
    ].tolist()

    closers = employee_table.loc[
        employee_table["can_close"].astype(bool),
        "employee_id",
    ].tolist()

    validate_staffing_pool(
        managers=managers,
        till_staff=till_staff,
        openers=openers,
        closers=closers,
        requirements=store_requirements,
    )

    for day in schedule_dates:
        model.Add(
            sum(
                work[(employee, day, shift)]
                for employee in employees
                for shift in SHIFTS
            )
            >= int(store_requirements["minimum_staff"])
        )

        model.Add(
            sum(
                work[(employee, day, shift)]
                for employee in managers
                for shift in SHIFTS
            )
            >= int(store_requirements["minimum_managers"])
        )

        model.Add(
            sum(
                work[(employee, day, shift)]
                for employee in till_staff
                for shift in SHIFTS
            )
            >= int(store_requirements["minimum_till_staff"])
        )

        model.Add(
            sum(
                work[(employee, day, shift)]
                for employee in openers
                for shift in ("OPEN", "SHORT_OPEN")
            )
            >= int(store_requirements["minimum_openers"])
        )

        model.Add(
            sum(
                work[(employee, day, shift)]
                for employee in closers
                for shift in ("CLOSE", "SHORT_CLOSE")
            )
            >= int(store_requirements["minimum_closers"])
        )


def validate_staffing_pool(
    *,
    managers: list[str],
    till_staff: list[str],
    openers: list[str],
    closers: list[str],
    requirements: dict[str, int],
) -> None:
    """Detect obviously impossible requirements before solving."""

    checks = {
        "managers": (
            len(managers),
            int(requirements["minimum_managers"]),
        ),
        "till-trained employees": (
            len(till_staff),
            int(requirements["minimum_till_staff"]),
        ),
        "qualified openers": (
            len(openers),
            int(requirements["minimum_openers"]),
        ),
        "qualified closers": (
            len(closers),
            int(requirements["minimum_closers"]),
        ),
    }

    for name, (available, required) in checks.items():
        if available < required:
            raise ValueError(
                f"The rota requires {required} {name}, "
                f"but only {available} are available in the employee data."
            )


# ==========================================================
# SOFT CONSTRAINTS
# ==========================================================

def add_soft_constraints(
    model: cp_model.CpModel,
    work: dict,
    employees: list[str],
    schedule_dates: list[pd.Timestamp],
    employees_df: pd.DataFrame,
    leave_df: pd.DataFrame,
) -> list:
    """Return all penalty expressions for the objective."""

    penalties = []

    add_requested_leave_penalties(
        penalties=penalties,
        work=work,
        schedule_dates=schedule_dates,
        leave_df=leave_df,
    )

    contract_groups = create_contract_groups(
        employees=employees,
        employees_df=employees_df,
    )

    add_fairness_spread_penalty(
        model=model,
        penalties=penalties,
        work=work,
        dates=schedule_dates,
        shifts=("CLOSE", "SHORT_CLOSE"),
        employee_groups=contract_groups,
        weight=CLOSING_FAIRNESS_WEIGHT,
        label="closing",
    )

    weekend_dates = [
        day
        for day in schedule_dates
        if day.weekday() in (5, 6)
    ]

    add_fairness_spread_penalty(
        model=model,
        penalties=penalties,
        work=work,
        dates=weekend_dates,
        shifts=tuple(SHIFTS),
        employee_groups=contract_groups,
        weight=WEEKEND_FAIRNESS_WEIGHT,
        label="weekend",
    )

    return penalties


def add_requested_leave_penalties(
    penalties: list,
    work: dict,
    schedule_dates: list[pd.Timestamp],
    leave_df: pd.DataFrame,
) -> None:
    """Avoid requested leave where possible."""

    requested = leave_df[
        leave_df["status"].astype(str).str.upper()
        == "REQUESTED"
    ]

    for row in requested.itertuples(index=False):
        employee = str(row.employee_id)
        leave_start = pd.Timestamp(
            row.start_date
        ).normalize()
        leave_end = pd.Timestamp(
            row.end_date
        ).normalize()

        for day in schedule_dates:
            if leave_start <= day <= leave_end:
                penalties.extend(
                    REQUESTED_LEAVE_WEIGHT
                    * work[(employee, day, shift)]
                    for shift in SHIFTS
                )


def create_contract_groups(
    employees: list[str],
    employees_df: pd.DataFrame,
) -> dict[int, list[str]]:
    """Group employees by contracted weekly hours."""

    contract_lookup = (
        employees_df
        .assign(
            employee_id=lambda frame:
            frame["employee_id"].astype(str)
        )
        .set_index("employee_id")["contract_hours"]
        .astype(int)
        .to_dict()
    )

    groups: dict[int, list[str]] = {}

    for employee in employees:
        groups.setdefault(
            contract_lookup[employee],
            [],
        ).append(employee)

    return groups


def add_fairness_spread_penalty(
    *,
    model: cp_model.CpModel,
    penalties: list,
    work: dict,
    dates: list[pd.Timestamp],
    shifts: tuple[str, ...],
    employee_groups: dict[int, list[str]],
    weight: int,
    label: str,
) -> None:
    """
    Penalise the difference between the highest and lowest
    number of selected duties in comparable employee groups.
    """

    if not dates:
        return

    maximum_count = len(dates)

    for contract_hours, group in employee_groups.items():
        if len(group) < 2:
            continue

        counts = {
            employee: sum(
                work[(employee, day, shift)]
                for day in dates
                for shift in shifts
            )
            for employee in group
        }

        group_max = model.NewIntVar(
            0,
            maximum_count,
            f"{label}_max_{contract_hours}",
        )
        group_min = model.NewIntVar(
            0,
            maximum_count,
            f"{label}_min_{contract_hours}",
        )

        for count in counts.values():
            model.Add(count <= group_max)
            model.Add(count >= group_min)

        spread = model.NewIntVar(
            0,
            maximum_count,
            f"{label}_spread_{contract_hours}",
        )

        model.Add(spread == group_max - group_min)

        penalties.append(weight * spread)


# ==========================================================
# SUPPORT FUNCTIONS
# ==========================================================

def group_dates_by_week(
    schedule_dates: list[pd.Timestamp],
) -> dict[tuple[int, int], list[pd.Timestamp]]:
    """Group dates using ISO year and ISO week."""

    groups: dict[
        tuple[int, int],
        list[pd.Timestamp],
    ] = {}

    for day in schedule_dates:
        iso = day.isocalendar()
        key = (int(iso.year), int(iso.week))
        groups.setdefault(key, []).append(day)

    return groups


def create_approved_leave_lookup(
    employees: list[str],
    leave_df: pd.DataFrame,
) -> dict[str, set[pd.Timestamp]]:
    """Create approved leave date sets for all employees."""

    lookup = {
        employee: set()
        for employee in employees
    }

    approved = leave_df[
        leave_df["status"].astype(str).str.upper()
        == "APPROVED"
    ]

    for row in approved.itertuples(index=False):
        employee = str(row.employee_id)

        if employee not in lookup:
            continue

        lookup[employee].update(
            pd.Timestamp(day).normalize()
            for day in pd.date_range(
                row.start_date,
                row.end_date,
                freq="D",
            )
        )

    return lookup


def extract_rota(
    *,
    solver: cp_model.CpSolver,
    work: dict,
    employees: list[str],
    schedule_dates: list[pd.Timestamp],
    employees_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert the solver solution into a normal table."""

    employee_information = (
        employees_df
        .assign(
            employee_id=lambda frame:
            frame["employee_id"].astype(str)
        )
        .set_index("employee_id")
        .to_dict("index")
    )

    rows = []

    for employee in employees:
        details = employee_information[employee]

        for day in schedule_dates:
            assigned_shift = "OFF"

            for shift in SHIFTS:
                if solver.Value(
                    work[(employee, day, shift)]
                ):
                    assigned_shift = shift
                    break

            if assigned_shift == "OFF":
                shift_time = "OFF"
                hours = 0
            else:
                shift = SHIFT_DETAILS[assigned_shift]
                shift_time = (
                    f"{shift['start']:02d}:00 - "
                    f"{shift['end']:02d}:00"
                )
                hours = shift["hours"]

            rows.append(
                {
                    "employee_id": employee,
                    "employee_name": details["employee_name"],
                    "role": details["role"],
                    "contract_hours": details["contract_hours"],
                    "contract_type": details["contract_type"],
                    "date": day,
                    "day_name": day.day_name(),
                    "shift": assigned_shift,
                    "shift_time": shift_time,
                    "hours": hours,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["date", "employee_name"]
        )
        .reset_index(drop=True)
    )


def failed_result(
    *,
    status: int,
    status_name: str,
    solver: cp_model.CpSolver,
) -> RotaResult:
    """Return a frontend-friendly result when solving fails."""

    if status == cp_model.INFEASIBLE:
        message = (
            "No valid rota can satisfy all current hard constraints. "
            "Check employee numbers, contracted hours, leave, "
            "availability and staffing requirements."
        )
    elif status == cp_model.MODEL_INVALID:
        message = (
            "The rota model is invalid and must be corrected."
        )
    else:
        message = (
            "The optimiser did not find a rota within the allowed time."
        )

    return RotaResult(
        success=False,
        status=status_name,
        message=message,
        solve_time_seconds=solver.WallTime(),
    )


def report_progress(
    callback: ProgressCallback | None,
    message: str,
) -> None:
    """Send progress messages to the frontend when supplied."""

    if callback is not None:
        callback(message)


# ==========================================================
# TEMPORARY BACKEND TEST
# ==========================================================

if __name__ == "__main__":
    def show_progress(message: str) -> None:
        print(message)

    result = generate_rota(
    input_file="DATA AND LEAVE.xlsx",
    start_date="2027-01-01",
    end_date="2027-12-31",
    output_file="output/generated_rota_2027.xlsx",
    maximum_solve_time=600,
    progress_callback=show_progress,
)
    print()
    print("Success:", result.success)
    print("Status:", result.status)
    print("Message:", result.message)
    print("Solve time:", result.solve_time_seconds)

    if result.output_file is not None:
        print("Saved to:", result.output_file)