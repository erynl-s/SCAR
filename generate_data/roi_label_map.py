# create a dictionary to map labels to structures

ROI_labels = {
    0: 'background',
    2: 'left cerebral white matter',
    3: 'left cerebral cortex',
    4: 'left lateral ventricle',
    5: 'left inferior lateral ventricle',
    7: 'left cerebellum white matter',
    8: 'left cerebellum cortex',
    10: 'left thalamus',
    11: 'left caudate',
    12: 'left putamen',
    13: 'left pallidum',
    14: '3rd ventricle',
    15: '4th ventricle',
    16: 'brain-stem',
    17: 'left hippocampus',
    18: 'left amygdala',
    26: 'left accumbens area',
    24: 'csf',
    28: 'left ventral DC',
    41: 'right cerebral white matter',
    42: 'right cerebral cortex',
    43: 'right lateral ventricle',
    44: 'right inferior lateral ventricle',
    46: 'right cerebellum white matter',
    47: 'right cerebellum cortex',
    49: 'right thalamus',
    50: 'right caudate',
    51: 'right putamen',
    52: 'right pallidum',
    53: 'right hippocampus',
    54: 'right amygdala',
    58: 'right accumbens area',
    60: 'right ventral DC'}


def get_numerical_label(structure_name):
    for label, name in ROI_labels.items():
        if name == structure_name:
            return label
    return "ERROR: invalid label"

original_volumes = {
    'total intracranial':1456440.5,
    'left cerebral white matter':232542.08,
    'left cerebral cortex':230194.69,
    'left lateral ventricle':12899.493,
    'left inferior lateral ventricle':732.627,
    'left cerebellum white matter':16725.748,
    'left cerebellum cortex':42575.91,
    'left thalamus':6709.93,
    'left caudate':4294.881,
    'left putamen':5218.082,
    'left pallidum':1595.237,
    '3rd ventricle':1229.109,
    '4th ventricle':1932.747,
    'brain-stem':19004.096,
    'left hippocampus':4028.992,
    'left amygdala':1813.628,
    'csf':305348.22,
    'left accumbens area':645.995,
    'left ventral DC':3985.661,
    'right cerebral white matter':233743.62,
    'right cerebral cortex':228660.34,
    'right lateral ventricle':14409.578,
    'right inferior lateral ventricle':891.757,
    'right cerebellum white matter':16493.105,
    'right cerebellum cortex':42355.152,
    'right thalamus':6878.698,
    'right caudate':4331.768,
    'right putamen':5390.772,
    'right pallidum':1590.361,
    'right hippocampus':3938.367,
    'right amygdala':1736.32,
    'right accumbens area':612.569,
    'right ventral DC':3931.05
}