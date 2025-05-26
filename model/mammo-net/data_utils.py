embed_data_dir = '/vol/biomedic3/data/EMBED/images/png/1024x768'
embed_counterfactuals_dir = '/vol/biomedic3/bglocker/ugproj/vg521/counterfactuals/cf-density-data'
embed_csv_dir = '/vol/biomedic3/mb121/tech-demo/code_for_demo/joined_simple.csv'

domain_maps = {
    "HOLOGIC, Inc.": 0,
    "GE MEDICAL SYSTEMS": 1,
    "FUJIFILM Corporation": 2,
    "GE HEALTHCARE": 3,
    "Lorad, A Hologic Company": 4,
}

tissue_maps = {"A": 0, "B": 1, "C": 2, "D": 3}

modelname_map = {
    "Selenia Dimensions": 0,
    "Senographe Essential VERSION ADS_53.40": 5,
    "Senographe Essential VERSION ADS_54.10": 5,
    "Senograph 2000D ADS_17.4.5": 2,
    "Senograph 2000D ADS_17.5": 2,
    "Lorad Selenia": 3,
    "Clearview CSm": 1,
    "Senographe Pristina": 4,
}

csv_error = (
    'For running EMBED code, you need to first generate the CSV file used for this study.\n'
    'You can do this by running:\n'
    'csv_generation_code/generate_embed_csv.ipynb'
)

embed_density_proportions = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.7}