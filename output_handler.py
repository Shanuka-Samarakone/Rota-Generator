from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_rota_to_excel(
    rota: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save the generated rota to a new Excel workbook.

    The workbook contains:
    - Generated_Rota
    - Weekly_View
    - Daily_Staffing
    """

    path = Path(output_path).expanduser().resolve()

    if path.suffix.lower() != ".xlsx":
        path = path.with_suffix(".xlsx")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_rota_output(rota)

    output = rota.copy()

    output["date"] = pd.to_datetime(
        output["date"],
        errors="raise",
    )

    weekly_view = create_weekly_view(output)
    daily_staffing = create_daily_staffing_summary(output)

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


def validate_rota_output(
    rota: pd.DataFrame,
) -> None:
    """
    Check that the backend produced the columns needed
    for the Excel output.
    """

    required_columns = {
        "employee_id",
        "employee_name",
        "role",
        "contract_hours",
        "contract_type",
        "date",
        "day_name",
        "shift",
        "shift_time",
        "hours",
    }

    missing_columns = sorted(
        required_columns - set(rota.columns)
    )

    if missing_columns:
        raise ValueError(
            "The generated rota is missing these columns: "
            + ", ".join(missing_columns)
        )

    if rota.empty:
        raise ValueError(
            "The generated rota is empty and cannot be exported."
        )


def create_weekly_view(
    rota: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a table with employees in rows and dates in columns.
    """

    weekly_data = rota.copy()

    weekly_data["display_shift"] = weekly_data.apply(
        lambda row: (
            "OFF"
            if row["shift"] == "OFF"
            else f"{row['shift']} ({row['shift_time']})"
        ),
        axis=1,
    )

    weekly_view = weekly_data.pivot(
        index=[
            "employee_id",
            "employee_name",
            "role",
            "contract_hours",
        ],
        columns="date",
        values="display_shift",
    )

    weekly_view = weekly_view.reset_index()

    weekly_view.columns = [
        column.strftime("%d-%m-%Y")
        if isinstance(column, pd.Timestamp)
        else column
        for column in weekly_view.columns
    ]

    return weekly_view


def create_daily_staffing_summary(
    rota: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a summary showing how many employees work each day
    and the total scheduled hours.
    """

    summary = rota.copy()

    summary["is_working"] = (
        summary["shift"] != "OFF"
    )

    daily_staffing = (
        summary
        .groupby(
            [
                "date",
                "day_name",
            ],
            as_index=False,
        )
        .agg(
            staff_working=(
                "is_working",
                "sum",
            ),
            scheduled_hours=(
                "hours",
                "sum",
            ),
        )
    )

    daily_staffing["date"] = (
        daily_staffing["date"]
        .dt.strftime("%d-%m-%Y")
    )

    return daily_staffing


def format_output_workbook(
    writer: pd.ExcelWriter,
) -> None:
    """
    Apply simple formatting to every worksheet.
    """

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

            maximum_length = max(
                len(value)
                for value in values
            )

            adjusted_width = min(
                maximum_length + 2,
                40,
            )

            column_letter = (
                column_cells[0].column_letter
            )

            worksheet.column_dimensions[
                column_letter
            ].width = adjusted_width