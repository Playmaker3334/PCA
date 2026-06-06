"""Mapeo centralizado dataset -> campos de texto y etiqueta.
Fuente unica de verdad para 01_extract, 04_evaluate y 05_baselines.
"""

# Campos en orden de prioridad. El primero que exista y sea str no vacio se usa.
DATASET_FIELDS = {
    "alpaca": ["instruction"],
    "triviaqa": ["question"],
    "mmlu": ["question"],
    "gsm8k": ["question"],
    "advbench": ["prompt", "goal", "behavior"],
    "harmbench": ["behavior", "prompt", "goal"],
    "jailbreakbench": ["Goal", "prompt", "goal", "behavior"],
    "xstest": ["prompt"],
}

DATASET_LABEL = {
    "alpaca": "safe",
    "triviaqa": "safe",
    "mmlu": "safe",
    "gsm8k": "safe",
    "xstest": "safe",
    "advbench": "unsafe",
    "harmbench": "unsafe",
    "jailbreakbench": "unsafe",
}

SAFE_CORPORA = [k for k, v in DATASET_LABEL.items() if v == "safe"]
UNSAFE_CORPORA = [k for k, v in DATASET_LABEL.items() if v == "unsafe"]


def get_text(example, dataset_name, *, allow_fallback=False):
    """Extrae el texto del campo correcto. Sin fallback permisivo por defecto.

    Devuelve None si no encuentra un campo valido (en vez de adivinar),
    lo que evita capturar opciones de respuesta en MMLU u otros campos.
    """
    fields = DATASET_FIELDS.get(dataset_name, [])
    for f in fields:
        v = example.get(f)
        if isinstance(v, str) and len(v.strip()) > 0:
            return v.strip()
    if allow_fallback:
        for v in example.values():
            if isinstance(v, str) and len(v.strip()) > 5:
                return v.strip()
    return None