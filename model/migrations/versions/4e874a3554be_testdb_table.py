"""testDB table

Revision ID: 4e874a3554be
Revises: fefb05158e6c
Create Date: 2022-07-31 20:56:02.463577

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4e874a3554be'
down_revision = 'fefb05158e6c'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user_commitment',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('esa_action_count', sa.Integer(), nullable=True),
    sa.Column('esa_total_count', sa.Integer(), nullable=True),
    sa.Column('auc_understanding_action_count', sa.Integer(), nullable=True),
    sa.Column('auc_understanding_total_count', sa.Integer(), nullable=True),
    sa.Column('auc_commitments_action_count', sa.Integer(), nullable=True),
    sa.Column('auc_commitments_total_count', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.drop_column('model_session', 'a11_count')
    op.drop_column('model_session', 'b09_count')
    op.drop_column('model_session', 'b05_count')
    op.drop_column('model_session', 'a05_count')
    op.drop_column('model_session', 'c09_count')
    op.drop_column('model_session', 'e04_count')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('model_session', sa.Column('e04_count', sa.INTEGER(), nullable=True))
    op.add_column('model_session', sa.Column('c09_count', sa.INTEGER(), nullable=True))
    op.add_column('model_session', sa.Column('a05_count', sa.INTEGER(), nullable=True))
    op.add_column('model_session', sa.Column('b05_count', sa.INTEGER(), nullable=True))
    op.add_column('model_session', sa.Column('b09_count', sa.INTEGER(), nullable=True))
    op.add_column('model_session', sa.Column('a11_count', sa.INTEGER(), nullable=True))
    op.drop_table('user_commitment')
    # ### end Alembic commands ###
