import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))

from application import *
from bureau import *
from credit_card_balance import *
from installements_payments import *
from merge_all_tables import *
from pos_cash import *
from previous_application import *
