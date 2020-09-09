import numpy as np
import os

class CONFIG:
    """
    To configure fail solve penalty, needs to be small enough that it doesn't distort reward,
    but large enough to make the agent avoid taking actions that result in fail-solve
    """
    def __init__(self, config_number):
        self.config_number = config_number

    def get_config(self):
        if self.config_number == 0:  # ChemSepExample LuybenPart
            # currently from thesis code, just copying iso & n butane to be same values
            compound_names = ["Ethane", "Propane", "Isobutane", "Butane",  "Iso-Pentane", "N-pentane"]
            Molar_weights = np.array([30.07, 44.097, 58.124, 58.124, 72.151, 72.151]) #g/mol
            Heating_value = np.array([51.9, 50.4, 49.5, 49.4,   55.2, 55.2]) #MJ/kg
            Price_per_MBTU = np.array([2.54, 4.27, 5.79, 5.31,  10.41, 10.41])  #  $/Million Btu
            MJ_per_MBTU = 1055.06
            # units $/MBTU * MBTU/MJ * MJ/kg * kg/g * g/mol  = $/mol
            sales_prices = Price_per_MBTU/MJ_per_MBTU * Heating_value * (Molar_weights/1000)  # now in $/mol
            required_purity = 0.95
            COCO_file = os.path.join(os.getcwd(), "Env\LuybenExamplePart.fsd")
            fail_solve_penalty = 0.5
            standard_args = COCO_file, sales_prices, fail_solve_penalty, required_purity
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
            fail_solve_penalty = 0.5
            standard_args = COCO_file, sales_prices, fail_solve_penalty
            return standard_args

        elif self.config_number == 2:  # loosely based on Luyben hydrocarbon example
            compound_names = ["Methane", "Ethane", "Propane", "Isobutane", "N-butane", "Iso-pentane", "N-pentane", "N-hexane", "n-heptane", "Nitrogen"]
            Heating_value = np.array([55.6, 51.9, 50.4, 49.5, 49.4, 55.2, 55.2, 47.7, 48.5])  # MJ/kg
            Price_per_MBTU = np.array([2.83, 2.54, 4.27, 5.79, 5.31, 10.41, 10.41, 10.41, 10.41])  # $/Million Btu
            Molar_weights = np.array([16.043, 30.07, 44.097, 58.124, 58.124, 72.151, 72.151, 86.18, 100.21])  # g/mol
            print("Haven't yet checked Iso-butane, N-hexane and n-heptane prices, assuming same as pentane here")
            MJ_per_MBTU = 1055.06
            sales_prices = Price_per_MBTU / MJ_per_MBTU * Heating_value * (Molar_weights / 1000)  # now in $/mol
            sales_prices = np.concatenate((sales_prices, [0]))  # nitrogen
            COCO_file = os.path.join(os.getcwd(), "Env\LuybenExampleFull.fsd")
            fail_solve_penalty = 3
            standard_args = COCO_file, sales_prices, fail_solve_penalty
            return standard_args

        elif self.config_number == 3:  # ChemSep Benzene Toluene P-xylene example
            compound_names = ["Benzene", "Toluene", "P-xylene"]
            molar_mass = np.array([78.11, 92.14, 106.16])  # g/mol
            # note MT is metric tonne
            # $/MT to $/g (T/1e3kg * 1e3kg/g)
            sales_prices_per_g = np.array([488.0, 488.0, 510.0]) / (1e3 * 1e3) # https://www.echemi.com/productsInformation/pid_Seven2868-benzene.html
            sales_prices = sales_prices_per_g * molar_mass # now in $/mol
            COCO_file = os.path.join(os.getcwd(), "Env\Benzene_Toluene_P_xylene.fsd")
            fail_solve_penalty = 0.5
            required_purity = 0.95
            standard_args = COCO_file, sales_prices, fail_solve_penalty, required_purity
            return standard_args

        elif self.config_number == 4: # Air seperation unit
            compound_names = ["Nitrogen", "Oxygen", "Argon"]
            #https: // en.wikipedia.org / wiki / Prices_of_chemical_elements
            molar_mass = np.array([14, 16, 40])
            sales_prices_per_g = np.array([0.140, 0.154, 0.931])/1000  #$/g
            sales_prices = sales_prices_per_g * molar_mass # now in $/mol
            COCO_file = os.path.join(os.getcwd(), "Env\ASU.fsd")
            fail_solve_penalty = 0.5
            required_purity = 0.985
            standard_args = COCO_file, sales_prices, fail_solve_penalty, required_purity
            return standard_args


