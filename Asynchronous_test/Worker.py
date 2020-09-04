import comtypes.client
import comtypes.gen
from comtypes import COMError
from comtypes import CoInitialize
import numpy as np
import time
import os

# tell comtypes to load type libs
cofeTlb = ('{0D1006C7-6086-4838-89FC-FBDCC0E98780}', 1, 0)  # COFE type lib
cofeTypes = comtypes.client.GetModule(cofeTlb)
coTlb = ('{4A5E2E81-C093-11D4-9F1B-0010A4D198C2}', 1, 1)  # CAPE-OPEN v1.1 type lib
coTypes = comtypes.client.GetModule(coTlb)


class Worker:
    """
    Worker that runs solves for a random column configuration
    """
    def __init__(self, global_counter, solve_time_list, total_steps=50, COCO_doc_path=os.path.join(os.getcwd(), "LuybenExamplePart.fsd")):
        self.total_steps = total_steps
        self.global_counter = global_counter
        self.global_count = 0
        self.solve_time_list = solve_time_list
        self.doc_path = COCO_doc_path
        self.quantity_basis = "mole"

    def setup(self):
        # have to do this outside of __init__ to prevent thread errors
        CoInitialize()
        self.doc = comtypes.client.CreateObject('COCO_COFE.Document', interface=cofeTypes.ICOFEDocument)

    def import_file(self):
        self.doc.Import(self.doc_path)

    def run(self):
        self.setup()
        while self.global_count < self.total_steps:
            self.run_step()

    def run_step(self):
        start_time = time.time()
        self.import_file()  # reset column to prevent errors from previous solve results
        # for this example we just set the column configuration to something random
        n_stages = np.random.uniform(20.0, 30.0)
        reflux_ratio = np.random.uniform(0.1, 2.0)
        reboil_ratio = np.random.uniform(0.1, 2.0)
        self.set_unit_inputs(n_stages, reflux_ratio, reboil_ratio)
        self.solve()
        solve_time = time.time() - start_time
        self.solve_time_list.append(solve_time)
        self.global_count = next(self.global_counter)
        print(f"global counter at step {self.global_count}\n")
        return

    def set_unit_inputs(self, n_stages, reflux_ratio, reboil_ratio):
        self.doc.GetUnit('Column_1').QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection).Item(
            "Number of stages").QueryInterface(coTypes.ICapeParameter).value = float(n_stages)
        self.doc.GetUnit('Column_1').QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection).Item("Feed1 stage").QueryInterface(coTypes.ICapeParameter).value = \
            float(round(n_stages/2))  # put the feed in the middle
        self.doc.GetUnit('Column_1').QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection).Item(
            "Reflux ratio").QueryInterface(coTypes.ICapeParameter).value = float(reflux_ratio)
        self.doc.GetUnit('Column_1').QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection).Item(
            "Reboil ratio").QueryInterface(coTypes.ICapeParameter).value = float(reboil_ratio)

    def solve(self):
        try:
            self.doc.Solve()
        except COMError as err:
            print(err)
