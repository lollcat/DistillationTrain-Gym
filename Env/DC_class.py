import comtypes.client
import comtypes.gen
from comtypes import COMError
from comtypes.automation import VARIANT
from comtypes import CoInitialize
import array
import numpy as np

# tell comtypes to load type libs
cofeTlb = ('{0D1006C7-6086-4838-89FC-FBDCC0E98780}', 1, 0)  # COFE type lib
cofeTypes = comtypes.client.GetModule(cofeTlb)
coTlb = ('{4A5E2E81-C093-11D4-9F1B-0010A4D198C2}', 1, 1)  # CAPE-OPEN v1.1 type lib
coTypes = comtypes.client.GetModule(coTlb)


class SimulatorDC:
    def __init__(self, doc_path):
        """
        doc_path: path to flowsheet.fsd COCO file
        :param doc_path:
        """
        self.doc_path = doc_path
        self.quantity_basis = "mole"
        CoInitialize()
        self.doc = comtypes.client.CreateObject('COCO_COFE.Document', interface=cofeTypes.ICOFEDocument)
        self.import_file()
        self.original_conditions = self.get_inlet_stream_conditions()
        self.max_pressure = self.doc.GetUnit('Valve_1').QueryInterface(
            coTypes.ICapeUtilities).Parameters.QueryInterface(coTypes.ICapeCollection).Item(
            "Pressure").QueryInterface(coTypes.ICapeParameter).value
        self.compound_names = self.doc.GetStream('1').QueryInterface(coTypes.ICapeThermoCompounds).GetCompoundList()[0]

    def import_file(self):
        self.doc.Import(self.doc_path)

    def set_inlet_stream(self, flows, temperature, pressure):
        try:
            self.doc.GetStream('1').QueryInterface(coTypes.ICapeThermoMaterial). \
                SetOverallProp("flow", self.quantity_basis, array.array('d', flows))
            self.doc.GetStream('1').QueryInterface(coTypes.ICapeThermoMaterial). \
                SetOverallProp("temperature", "", array.array('d', (temperature,)))
            self.doc.GetStream('1').QueryInterface(coTypes.ICapeThermoMaterial). \
                SetOverallProp("pressure", "", array.array('d', (pressure,)))

        except COMError:
            print("Error:", self.doc.GetStream('1').QueryInterface(coTypes.ECapeRoot).name)

    def get_inlet_stream_conditions(self):
        conditions = {
                "flows": np.array(self.doc.GetStream('1').QueryInterface(coTypes.ICapeThermoMaterial).
                    GetOverallProp("flow", self.quantity_basis)),
                "temperature": self.doc.GetStream('1').QueryInterface(coTypes.ICapeThermoMaterial).
                    GetOverallProp("temperature", "")[0],
                "pressure": self.doc.GetStream('1').QueryInterface(coTypes.ICapeThermoMaterial).
                    GetOverallProp("pressure", "")[0]
                    }
        return conditions

    def get_outlet_info(self):
        tops_flows = np.array(self.doc.GetStream('2').QueryInterface(coTypes.ICapeThermoMaterial).GetOverallProp("flow",
                                                                                                           self.quantity_basis))
        tops_temperature = \
            self.doc.GetStream('2').QueryInterface(coTypes.ICapeThermoMaterial).GetOverallProp("temperature", "")[0]
        tops_pressure = \
            self.doc.GetStream('2').QueryInterface(coTypes.ICapeThermoMaterial).GetOverallProp("pressure", "")[0]

        bottoms_flows = np.array(self.doc.GetStream('3').QueryInterface(coTypes.ICapeThermoMaterial).GetOverallProp("flow",
                                                                                                              self.quantity_basis))
        bottoms_temperature = \
            self.doc.GetStream('3').QueryInterface(coTypes.ICapeThermoMaterial).GetOverallProp("temperature", "")[0]
        bottoms_pressure = \
            self.doc.GetStream('3').QueryInterface(coTypes.ICapeThermoMaterial).GetOverallProp("pressure", "")[0]

        return [tops_flows, tops_temperature, tops_pressure], [bottoms_flows, bottoms_temperature,
                                                                  bottoms_pressure]

    def get_outputs(self):
        tops_flow = self.doc.GetStream('2').QueryInterface(coTypes.ICapeThermoMaterial).GetOverallProp("flow", self.quantity_basis)
        bottoms_flow = self.doc.GetStream('3').QueryInterface(coTypes.ICapeThermoMaterial).GetOverallProp("flow",
                                                                                                          self.quantity_basis)
        TAC = self.doc.GetUnit('Column_1').QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection). \
            Item("Total Annual Cost").QueryInterface(coTypes.ICapeParameter).value
        condenser_duty = self.doc.GetUnit('Column_1').QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection). \
            Item("Reboiler duty").QueryInterface(coTypes.ICapeParameter)
        reboiler_duty = self.doc.GetUnit('Column_1').QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection). \
            Item("Reboiler duty").QueryInterface(coTypes.ICapeParameter)
        return TAC, condenser_duty, reboiler_duty

    def get_unit_inputs(self):
        n_stages = self.doc.GetUnit('Column_1').QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection).Item(
            "Number of stages").QueryInterface(coTypes.ICapeParameter).value
        reflux_ratio = self.doc.GetUnit('Column_1').QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection).Item(
            "Reflux ratio").QueryInterface(coTypes.ICapeParameter).value
        reboil_ratio = self.doc.GetUnit('Column_1').QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection).Item(
            "Reboil ratio").QueryInterface(coTypes.ICapeParameter).value
        pressure_drop = self.doc.GetUnit("Vale_1").QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection).Item(
            "Pressure difference").QueryInterface(coTypes.ICapeParameter).value
        return n_stages, reflux_ratio, reboil_ratio, pressure_drop

    def set_unit_inputs(self, n_stages, reflux_ratio, reboil_ratio, pressure_drop_ratio):
        column_pressure = self.get_inlet_stream_conditions()["pressure"]*(1-pressure_drop_ratio)
        assert column_pressure > 0

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
        self.doc.GetUnit("Valve_1").QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection).Item(
            "Pressure").QueryInterface(coTypes.ICapeParameter).value = float(column_pressure)
        # also need to update column pressure to fit column input stream
        self.doc.GetUnit('Column_1').QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection).Item(
            "Condenser pressure").QueryInterface(coTypes.ICapeParameter).value = float(column_pressure)
        self.doc.GetUnit('Column_1').QueryInterface(coTypes.ICapeUtilities).Parameters.QueryInterface(
            coTypes.ICapeCollection).Item(
            "Top pressure").QueryInterface(coTypes.ICapeParameter).value = float(column_pressure)

    def solve(self):
        try:
            self.doc.Solve()
            return True  # 0 for sucess
        except COMError as err:
            #print(err)   #  for now let's not print as number of errors is stored in the DC_gym
            return False  # for failure

    def reset_flowsheet(self):
        self.set_inlet_stream(self.original_conditions["flows"], self.original_conditions["temperature"],
                              self.original_conditions["pressure"])
