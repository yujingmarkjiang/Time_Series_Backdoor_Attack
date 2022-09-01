UNIVARIATE_DATASET_NAMES = ['BirdChicken', 'ECG5000', 'Earthquakes', 'ElectricDevices', 'Haptics', 'PowerCons',
                            'ShapeletSim', 'Wine']

UNIVARIATE_DATASET_NAMES_2018 = ['BirdChicken', 'ECG5000', 'Earthquakes', 'ElectricDevices', 'Haptics', 'PowerCons',
                                 'ShapeletSim', 'Wine']

MTS_DATASET_NAMES = ['ArabicDigits', 'ECG', 'KickvsPunch', 'NetFlow', 'UWave']

UNIVARIATE_TEST = ['BirdChicken',  #Image
                   'ECG5000',  #ECG
                   'Earthquakes',  #Sensor
                   'ElectricDevices',  #Device
                   'Haptics',  #Motion
                   'PowerCons',  #Power
                   'ShapeletSim',  #Simulated
                   'Wine'  #Spectro
                   ]

MTS_TEST = ['ArabicDigits',
            'ECG',
            'KickvsPunch',
            'NetFlow',
            'UWave']

ITERATIONS = 1  # nb of random runs for random initializations

ARCHIVE_NAMES = ['UCRArchive_2018', 'mts_archive']

dataset_names_for_archive = {'UCRArchive_2018': UNIVARIATE_TEST,
                             'mts_archive': MTS_TEST}

CLASSIFIERS = ['fcn', 'resnet']

dataset_types = {'ElectricDevices': 'DEVICE', 'FordB': 'SENSOR',
                 'FordA': 'SENSOR', 'NonInvasiveFatalECG_Thorax2': 'ECG',
                 'NonInvasiveFatalECG_Thorax1': 'ECG', 'PhalangesOutlinesCorrect': 'IMAGE',
                 'HandOutlines': 'IMAGE', 'StarLightCurves': 'SENSOR',
                 'wafer': 'SENSOR', 'Two_Patterns': 'SIMULATED',
                 'UWaveGestureLibraryAll': 'MOTION', 'uWaveGestureLibrary_Z': 'MOTION',
                 'uWaveGestureLibrary_Y': 'MOTION', 'uWaveGestureLibrary_X': 'MOTION',
                 'Strawberry': 'SPECTRO', 'ShapesAll': 'IMAGE',
                 'ProximalPhalanxOutlineCorrect': 'IMAGE', 'MiddlePhalanxOutlineCorrect': 'IMAGE',
                 'DistalPhalanxOutlineCorrect': 'IMAGE', 'FaceAll': 'IMAGE',
                 'ECG5000': 'ECG', 'SwedishLeaf': 'IMAGE', 'ChlorineConcentration': 'SIMULATED',
                 '50words': 'IMAGE', 'ProximalPhalanxTW': 'IMAGE', 'ProximalPhalanxOutlineAgeGroup': 'IMAGE',
                 'MiddlePhalanxOutlineAgeGroup': 'IMAGE', 'DistalPhalanxTW': 'IMAGE',
                 'DistalPhalanxOutlineAgeGroup': 'IMAGE', 'MiddlePhalanxTW': 'IMAGE',
                 'Cricket_Z': 'MOTION', 'Cricket_Y': 'MOTION',
                 'Cricket_X': 'MOTION', 'Adiac': 'IMAGE',
                 'MedicalImages': 'IMAGE', 'SmallKitchenAppliances': 'DEVICE',
                 'ScreenType': 'DEVICE', 'RefrigerationDevices': 'DEVICE',
                 'LargeKitchenAppliances': 'DEVICE', 'Earthquakes': 'SENSOR',
                 'yoga': 'IMAGE', 'synthetic_control': 'SIMULATED',
                 'WordsSynonyms': 'IMAGE', 'Computers': 'DEVICE',
                 'InsectWingbeatSound': 'SENSOR', 'Phoneme': 'SENSOR',
                 'OSULeaf': 'IMAGE', 'FacesUCR': 'IMAGE',
                 'WormsTwoClass': 'MOTION', 'Worms': 'MOTION',
                 'FISH': 'IMAGE', 'Haptics': 'MOTION',
                 'Epilepsy': 'HAR', 'Ham': 'SPECTRO',
                 'Plane': 'SENSOR', 'InlineSkate': 'MOTION',
                 'Trace': 'SENSOR', 'ECG200': 'ECG',
                 'Lighting7': 'SENSOR', 'ItalyPowerDemand': 'SENSOR',
                 'Herring': 'IMAGE', 'Lighting2': 'SENSOR',
                 'Car': 'SENSOR', 'Meat': 'SPECTRO',
                 'Wine': 'SPECTRO', 'MALLAT': 'SIMULATED',
                 'Gun_Point': 'MOTION', 'CinC_ECG_torso': 'ECG',
                 'ToeSegmentation1': 'MOTION', 'ToeSegmentation2': 'MOTION',
                 'ArrowHead': 'IMAGE', 'OliveOil': 'SPECTRO',
                 'Beef': 'SPECTRO', 'CBF': 'SIMULATED',
                 'Coffee': 'SPECTRO', 'SonyAIBORobotSurfaceII': 'SENSOR',
                 'Symbols': 'IMAGE', 'FaceFour': 'IMAGE',
                 'ECGFiveDays': 'ECG', 'TwoLeadECG': 'ECG',
                 'BirdChicken': 'IMAGE', 'BeetleFly': 'IMAGE',
                 'ShapeletSim': 'SIMULATED', 'MoteStrain': 'SENSOR',
                 'SonyAIBORobotSurface': 'SENSOR', 'DiatomSizeReduction': 'IMAGE'}

themes_colors = {'IMAGE': 'red', 'SENSOR': 'blue', 'ECG': 'green',
                 'SIMULATED': 'yellow', 'SPECTRO': 'orange',
                 'MOTION': 'purple', 'DEVICE': 'gray'}
