"""add provenance-aware menu nutrition facts

Revision ID: 0017
Revises: 0016
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0017"
down_revision: Union[str, Sequence[str], None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for name in (
        "energy_kcal",
        "protein_g",
        "carbohydrates_g",
        "fat_g",
        "fiber_g",
        "nutrition_serving_g",
    ):
        op.add_column("menu_items", sa.Column(name, sa.Numeric(8, 2), nullable=True))
    op.add_column("menu_items", sa.Column("nutrition_source", sa.Text(), nullable=True))
    op.add_column(
        "menu_items",
        sa.Column(
            "nutrition_status",
            sa.Text(),
            server_default="UNKNOWN",
            nullable=False,
        ),
    )
    op.add_column(
        "menu_items",
        sa.Column("nutrition_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_menu_items_nutrition_status",
        "menu_items",
        "nutrition_status IN ('UNKNOWN', 'ESTIMATED', 'VERIFIED')",
    )
    op.create_check_constraint(
        "ck_menu_items_nutrition_values_nonnegative",
        "menu_items",
        "energy_kcal IS NULL OR energy_kcal >= 0",
    )
    for name in ("protein_g", "carbohydrates_g", "fat_g", "fiber_g"):
        op.create_check_constraint(
            f"ck_menu_items_{name}_nonnegative",
            "menu_items",
            f"{name} IS NULL OR {name} >= 0",
        )
    op.create_check_constraint(
        "ck_menu_items_nutrition_serving_positive",
        "menu_items",
        "nutrition_serving_g IS NULL OR nutrition_serving_g > 0",
    )
    op.create_check_constraint(
        "ck_menu_items_unknown_nutrition_empty",
        "menu_items",
        "nutrition_status <> 'UNKNOWN' OR ("
        "energy_kcal IS NULL AND protein_g IS NULL AND carbohydrates_g IS NULL AND "
        "fat_g IS NULL AND fiber_g IS NULL AND nutrition_serving_g IS NULL AND "
        "nutrition_source IS NULL"
        ")",
    )
    op.create_check_constraint(
        "ck_menu_items_estimated_nutrition_source",
        "menu_items",
        "nutrition_status <> 'ESTIMATED' OR ("
        "(energy_kcal IS NOT NULL OR protein_g IS NOT NULL OR carbohydrates_g IS NOT NULL OR "
        "fat_g IS NOT NULL OR fiber_g IS NOT NULL) AND nutrition_source IS NOT NULL AND "
        "length(trim(nutrition_source)) > 0"
        ")",
    )
    op.create_check_constraint(
        "ck_menu_items_verified_nutrition_complete",
        "menu_items",
        "nutrition_status <> 'VERIFIED' OR ("
        "energy_kcal IS NOT NULL AND protein_g IS NOT NULL AND "
        "carbohydrates_g IS NOT NULL AND fat_g IS NOT NULL AND fiber_g IS NOT NULL AND "
        "nutrition_serving_g IS NOT NULL AND nutrition_source IS NOT NULL AND "
        "length(trim(nutrition_source)) > 0"
        ")",
    )


def downgrade() -> None:
    for name in (
        "ck_menu_items_verified_nutrition_complete",
        "ck_menu_items_estimated_nutrition_source",
        "ck_menu_items_unknown_nutrition_empty",
        "ck_menu_items_nutrition_serving_positive",
        "ck_menu_items_fiber_g_nonnegative",
        "ck_menu_items_fat_g_nonnegative",
        "ck_menu_items_carbohydrates_g_nonnegative",
        "ck_menu_items_protein_g_nonnegative",
        "ck_menu_items_nutrition_values_nonnegative",
        "ck_menu_items_nutrition_status",
    ):
        op.drop_constraint(name, "menu_items", type_="check")
    for name in (
        "nutrition_updated_at",
        "nutrition_status",
        "nutrition_source",
        "nutrition_serving_g",
        "fiber_g",
        "fat_g",
        "carbohydrates_g",
        "protein_g",
        "energy_kcal",
    ):
        op.drop_column("menu_items", name)
