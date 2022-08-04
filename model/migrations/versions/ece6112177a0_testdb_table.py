"""testDB table

Revision ID: ece6112177a0
Revises: 7bfbd3d0b3fd
Create Date: 2022-07-31 19:35:58.773397

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ece6112177a0'
down_revision = '7bfbd3d0b3fd'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('model_session', sa.Column('a04_count', sa.Integer(), nullable=True))
    op.add_column('model_session', sa.Column('a08_count', sa.Integer(), nullable=True))
    op.add_column('model_session', sa.Column('b04_count', sa.Integer(), nullable=True))
    op.add_column('model_session', sa.Column('b08_count', sa.Integer(), nullable=True))
    op.add_column('model_session', sa.Column('c09_count', sa.Integer(), nullable=True))
    op.add_column('model_session', sa.Column('e03_count', sa.Integer(), nullable=True))
    op.add_column('model_session', sa.Column('main_node_count', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('model_session', 'main_node_count')
    op.drop_column('model_session', 'e03_count')
    op.drop_column('model_session', 'c09_count')
    op.drop_column('model_session', 'b08_count')
    op.drop_column('model_session', 'b04_count')
    op.drop_column('model_session', 'a08_count')
    op.drop_column('model_session', 'a04_count')
    # ### end Alembic commands ###