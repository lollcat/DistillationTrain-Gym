import numpy as np
import os

class CONFIG:
    def __init__(self, config_number):
        self.config_number = config_number

    def get_config(self):
        if self.config_number == 0:  # ChemSepExample
            # currently from thesis code, just copying iso & n butane to be same values
            compound_names = ["Methane", "Ethane", "Propane", "Isobutane", "Butane",  "Iso-Pentane", "N-pentane"]
            Molar_weights = np.array([16.043, 30.07, 44.097, 58.124, 58.124, 72.151, 72.151]) #g/mol
            Heating_value = np.array([55.6,  51.9, 50.4, 49.5, 49.4,   55.2, 55.2]) #MJ/kg
            Price_per_MBTU = np.array([2.83, 2.54, 4.27, 5.79, 5.31,  10.41, 10.41])  #  $/Million Btu
            MJ_per_MBTU = 1055.06
            # units $/MBTU * MBTU/MJ * MJ/kg * kg/g * g/mol  = $/mol
            sales_prices = Price_per_MBTU/MJ_per_MBTU * Heating_value * (Molar_weights/1000)  # now in $/mol


            COCO_file = os.path.join(os.getcwd(), "Env\ChemSepExample.fsd")

            standard_args = COCO_file, sales_prices
            return standard_args

        elif self.config_number == 1:  # ThomsonKing example
            # currently from thesis code, just copying iso & n butane to be same values
            compound_names = ["Ethane", "Propane", "Butane", "N-Butane", "Propylene", "1-Butene", "N-pentane"]
            Molar_weights = np.array([30.07,  42.08,  44.097,  56.108,  58.124,  72.15])
            Heating_value = np.array([51.9,   49.0,   50.4,    48.5,    49.4,    48.6])  # MJ/kg
            Price_per_MBTU = np.array([2.54,  17.58,  4.27,    29.47,   5.31,    13.86])  # $/Million Btu
            MJ_per_MBTU = 1055.06
            sales_prices = Price_per_MBTU/MJ_per_MBTU * Heating_value * (Molar_weights/1000)  # now in $/mol

            COCO_file = os.path.join(os.getcwd(), "Env\ThomsonKing.fsd")

            standard_args = COCO_file, sales_prices
            return standard_args

        elif self.config_number == 2:  # Luyben hydrocarbon example
            order_checked = False
            assert order_checked  #COCO order matches this one"
            compound_names = ["Methane", "Ethane", "Propane", "Isobutane", "N-butane", "N-pentane", "N-hexane", "n-heptane", "Nitrogen"]
            Heating_value = np.array([55.6, 51.9, 50.4, 49.5, 49.4, 55.2, 47.7, 48.5])  # MJ/kg
            Price_per_MBTU = np.array([2.83, 2.54, 4.27, 5.79, 5.31, 10.41, 10.41, 10.41])  # $/Million Btu
            Molar_weights = np.array([16.043, 30.07, 44.097, 58.124, 58.124, 72.151, 86.18, 100.21])  # g/mol
            print("Haven't yet checked N-hexane and n-heptane prices, assuming same as pentane here")
            MJ_per_MBTU = 1055.06
            sales_prices = Price_per_MBTU / MJ_per_MBTU * Heating_value * (Molar_weights / 1000)  # now in $/mol
            sales_prices = np.concatenate((sales_prices, [0]))  # nitrogen
            COCO_file = os.path.join(os.getcwd(), "Env\LuybenExample.fsd")
            standard_args = COCO_file, sales_prices
            return standard_args