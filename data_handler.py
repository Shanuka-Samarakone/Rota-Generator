from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_SHEETS = {
    "Employees",
    "Employee_Department_Skills",
    "Leave",
    "Unavailability",
    "Store_Requirements",
}

REQUIRED_COLUMNS = {
    "Employees": {
        "employee_id",
        "employee_name",
        "role",
        "contract_hours",
        "contract_type",
        "till_trained",
        "can_open",
        "can_close",
    },
    "Employee_Department_Skills": {
        "employee_id",
        "department",
        "priority",
    },
    "Leave": {
        "employee_id",
        "start_date",
        "end_date",
        "status",
    },
    "Unavailability": {
        "employee_id",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    },
    "Store_Requirements": {
        "parameter",
        "value",
    },
}

WEEKDAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

BOOLEAN_COLUMNS = [
    "till_trained",
    "can_open",
    "can_close",
]


def load_excel_data(file_path: str | Path) -> dict[str, Any]:
    """
    Load, validate and clean the rota input workbook.

    The returned dictionary is designed to match backend.py.
    """

    path = Path(file_path).expanduser().resolve()

    validate_input_file(path)

    try:
        workbook = pd.ExcelFile(path, engine="openpyxl")
    except Exception as error:
        raise ValueError(
            f"The Excel workbook could not be opened: {error}"
        ) from error

    validate_sheet_names(workbook.sheet_names)

    raw_data = {
        sheet_name: pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            engine="openpyxl",
        )
        for sheet_name in REQUIRED_SHEETS
    }

    for sheet_name, dataframe in raw_data.items():
        raw_data[sheet_name] = clean_column_names(dataframe)
        validate_required_columns(
            raw_data[sheet_name],
            sheet_name,
        )

    employees = clean_employees(raw_data["Employees"])
    department_skills = clean_department_skills(
        raw_data["Employee_Department_Skills"]
    )
    leave = clean_leave(raw_data["Leave"])
    unavailability = clean_unavailability(
        raw_data["Unavailability"]
    )
    store_requirements = clean_store_requirements(
        raw_data["Store_Requirements"]
    )

    validate_cross_sheet_employee_ids(
        employees=employees,
        department_skills=department_skills,
        leave=leave,
        unavailability=unavailability,
    )

    return {
        "employees": employees,
        "department_skills": department_skills,
        "leave": leave,
        "unavailability": unavailability,
        "store_requirements": store_requirements,
    }


def validate_input_file(path: Path) -> None:
    """Check that the selected input file exists and is an .xlsx workbook."""

    if not path.exists():
        raise FileNotFoundError(
            f"The Excel file could not be found:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"The selected path is not a file:\n{path}"
        )

    if path.suffix.lower() != ".xlsx":
        raise ValueError(
            "The input file must be an Excel .xlsx workbook."
        )


def validate_sheet_names(sheet_names: list[str]) -> None:
    """Check that every required worksheet exists."""

    missing_sheets = sorted(
        REQUIRED_SHEETS - set(sheet_names)
    )

    if missing_sheets:
        raise ValueError(
            "The workbook is missing these required sheets: "
            + ", ".join(missing_sheets)
        )


def clean_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardise spreadsheet column names."""

    cleaned = dataframe.copy()

    cleaned.columns = (
        cleaned.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return cleaned


def validate_required_columns(
    dataframe: pd.DataFrame,
    sheet_name: str,
) -> None:
    """Check that one worksheet contains every required column."""

    required = REQUIRED_COLUMNS[sheet_name]
    missing_columns = sorted(
        required - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"The '{sheet_name}' sheet is missing these columns: "
            + ", ".join(missing_columns)
        )


def clean_employees(
    employees: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and validate the Employees worksheet."""

    cleaned = employees.copy()

    text_columns = [
        "employee_id",
        "employee_name",
        "role",
        "contract_type",
    ]

    for column in text_columns:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
        )

    if cleaned["employee_id"].isna().any():
        raise ValueError(
            "The Employees sheet contains a blank employee_id."
        )

    if cleaned["employee_name"].isna().any():
        raise ValueError(
            "The Employees sheet contains a blank employee_name."
        )

    cleaned["contract_hours"] = pd.to_numeric(
        cleaned["contract_hours"],
        errors="raise",
    )

    if (cleaned["contract_hours"] < 0).any():
        raise ValueError(
            "Employee contract hours cannot be negative."
        )

    for column in BOOLEAN_COLUMNS:
        cleaned[column] = cleaned[column].apply(
            convert_to_boolean
        )

    duplicate_ids = cleaned.loc[
        cleaned["employee_id"].duplicated(keep=False),
        "employee_id",
    ].dropna().unique()

    if len(duplicate_ids) > 0:
        raise ValueError(
            "Duplicate employee IDs were found in Employees: "
            + ", ".join(map(str, duplicate_ids))
        )

    return cleaned.reset_index(drop=True)


def clean_department_skills(
    department_skills: pd.DataFrame,
) -> pd.DataFrame:
    """Clean the Employee_Department_Skills worksheet."""

    cleaned = department_skills.copy()

    cleaned["employee_id"] = (
        cleaned["employee_id"]
        .astype("string")
        .str.strip()
    )

    cleaned["department"] = (
        cleaned["department"]
        .astype("string")
        .str.strip()
    )

    cleaned["priority"] = pd.to_numeric(
        cleaned["priority"],
        errors="raise",
    ).astype(int)

    if (cleaned["priority"] < 1).any():
        raise ValueError(
            "Department priority values must be 1 or greater."
        )

    return cleaned.reset_index(drop=True)


def clean_leave(
    leave: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and validate the Leave worksheet."""

    cleaned = leave.copy()

    cleaned["employee_id"] = (
        cleaned["employee_id"]
        .astype("string")
        .str.strip()
    )

    cleaned["start_date"] = pd.to_datetime(
        cleaned["start_date"],
        errors="raise",
    ).dt.normalize()

    cleaned["end_date"] = pd.to_datetime(
        cleaned["end_date"],
        errors="raise",
    ).dt.normalize()

    cleaned["status"] = (
        cleaned["status"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    valid_statuses = {
        "APPROVED",
        "REQUESTED",
        "REJECTED",
    }

    invalid_statuses = sorted(
        set(cleaned["status"].dropna())
        - valid_statuses
    )

    if invalid_statuses:
        raise ValueError(
            "The Leave sheet contains invalid statuses: "
            + ", ".join(map(str, invalid_statuses))
        )

    invalid_dates = (
        cleaned["end_date"] < cleaned["start_date"]
    )

    if invalid_dates.any():
        invalid_rows = (
            cleaned.index[invalid_dates] + 2
        ).tolist()

        raise ValueError(
            "Leave end_date occurs before start_date on Excel row(s): "
            + ", ".join(map(str, invalid_rows))
        )

    return cleaned.reset_index(drop=True)


def clean_unavailability(
    unavailability: pd.DataFrame,
) -> pd.DataFrame:
    """Clean recurring weekly unavailability values."""

    cleaned = unavailability.copy()

    cleaned["employee_id"] = (
        cleaned["employee_id"]
        .astype("string")
        .str.strip()
    )

    for weekday in WEEKDAYS:
        cleaned[weekday] = cleaned[weekday].apply(
            convert_to_boolean
        )

    duplicate_ids = cleaned.loc[
        cleaned["employee_id"].duplicated(keep=False),
        "employee_id",
    ].dropna().unique()

    if len(duplicate_ids) > 0:
        raise ValueError(
            "Duplicate employee IDs were found in Unavailability: "
            + ", ".join(map(str, duplicate_ids))
        )

    return cleaned.reset_index(drop=True)


def clean_store_requirements(
    requirements: pd.DataFrame,
) -> dict[str, int]:
    """Convert Store_Requirements into a dictionary."""

    cleaned = requirements.copy()

    cleaned["parameter"] = (
        cleaned["parameter"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    cleaned["value"] = pd.to_numeric(
        cleaned["value"],
        errors="raise",
    )

    if cleaned["parameter"].isna().any():
        raise ValueError(
            "Store_Requirements contains a blank parameter."
        )

    if cleaned["parameter"].duplicated().any():
        duplicates = cleaned.loc[
            cleaned["parameter"].duplicated(keep=False),
            "parameter",
        ].dropna().unique()

        raise ValueError(
            "Duplicate store requirement parameters were found: "
            + ", ".join(map(str, duplicates))
        )

    if (cleaned["value"] < 0).any():
        raise ValueError(
            "Store requirement values cannot be negative."
        )

    non_integer_values = (
        cleaned["value"] % 1 != 0
    )

    if non_integer_values.any():
        raise ValueError(
            "Store requirement values must be whole numbers."
        )

    return {
        str(parameter): int(value)
        for parameter, value in zip(
            cleaned["parameter"],
            cleaned["value"],
        )
    }


def validate_cross_sheet_employee_ids(
    employees: pd.DataFrame,
    department_skills: pd.DataFrame,
    leave: pd.DataFrame,
    unavailability: pd.DataFrame,
) -> None:
    """Ensure every employee ID used elsewhere exists in Employees."""

    known_ids = set(
        employees["employee_id"].dropna()
    )

    sheets = {
        "Employee_Department_Skills": department_skills,
        "Leave": leave,
        "Unavailability": unavailability,
    }

    for sheet_name, dataframe in sheets.items():
        referenced_ids = set(
            dataframe["employee_id"].dropna()
        )

        unknown_ids = sorted(
            referenced_ids - known_ids
        )

        if unknown_ids:
            raise ValueError(
                f"The '{sheet_name}' sheet contains unknown employee IDs: "
                + ", ".join(map(str, unknown_ids))
            )


def convert_to_boolean(value: Any) -> bool:
    """Convert common Excel values into a Python Boolean."""

    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False

    text = str(value).strip().lower()

    true_values = {
        "true",
        "yes",
        "y",
        "1",
    }

    false_values = {
        "false",
        "no",
        "n",
        "0",
        "",
    }

    if text in true_values:
        return True

    if text in false_values:
        return False

    raise ValueError(
        f"'{value}' could not be converted to True or False."
    )


def save_rota_to_excel(
    rota: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save the completed rota to Excel.

    The workbook contains:
    - Generated_Rota: full row-by-row schedule
    - Weekly_View: employee-by-date grid
    - Daily_Staffing: daily staff and hours summary
    """

    path = Path(output_path).expanduser().resolve()

    if path.suffix.lower() != ".xlsx":
        path = path.with_suffix(".xlsx")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_output_columns = {
        "employee_id",
        "employee_name",
        "date",
        "shift",
        "hours",
    }

    missing_columns = sorted(
        required_output_columns - set(rota.columns)
    )

    if missing_columns:
        raise ValueError(
            "The rota output is missing these columns: "
            + ", ".join(missing_columns)
        )

    output = rota.copy()
    output["date"] = pd.to_datetime(
        output["date"],
        errors="raise",
    )

    weekly_view = (
        output.pivot(
            index=[
                "employee_id",
                "employee_name",
            ],
            columns="date",
            values="shift",
        )
        .reset_index()
    )

    daily_staffing = (
        output.assign(
            is_working=output["shift"].ne("OFF")
        )
        .groupby(
            [
                "date",
                "day_name",
            ],
            as_index=False,
        )
        .agg(
            staff_working=("is_working", "sum"),
            scheduled_hours=("hours", "sum"),
        )
    )

    try:
        with pd.ExcelWriter(
            path,
            engine="openpyxl",
        ) as writer:
            output.to_excel(
                writer,
                sheet_name="Generated_Rota",
                index=False,
            )

            weekly_view.to_excel(
                writer,
                sheet_name="Weekly_View",
                index=False,
            )

            daily_staffing.to_excel(
                writer,
                sheet_name="Daily_Staffing",
                index=False,
            )

            format_output_workbook(writer)

    except PermissionError as error:
        raise PermissionError(
            "The output workbook could not be saved. "
            "Close it in Excel and try again."
        ) from error

    except Exception as error:
        raise IOError(
            f"The rota could not be saved: {error}"
        ) from error

    return path


def format_output_workbook(
    writer: pd.ExcelWriter,
) -> None:
    """Apply simple widths, filters and frozen headers."""

    for worksheet in writer.book.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column_cells in worksheet.columns:
            values = [
                str(cell.value)
                if cell.value is not None
                else ""
                for cell in column_cells
            ]

            maximum_length = min(
                max(len(value) for value in values) + 2,
                35,
            )

            column_letter = column_cells[0].column_letter
            worksheet.column_dimensions[
                column_letter
            ].width = maximum_length