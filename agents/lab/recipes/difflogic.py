"""Recipe: Deep Differentiable Logic Gate Networks (Petersen et al., 2022), MNIST."""
from __future__ import annotations

from pathlib import Path

NAME = "difflogic_mnist"
TITLE = "Deep Differentiable Logic Gate Networks"
ARXIV = "2210.08277"
FILES_DIR = Path(__file__).parent / "difflogic_files"
PRIMARY_METRIC = "accuracy_mnist"
FIELD = "edge and efficient ML"
FULL_DATASET_MINUTES = 10


def matches(title: str, arxiv_id: str, text: str) -> bool:
    t = (title + " " + text[:4000]).lower()
    return (
        arxiv_id.startswith(ARXIV)
        or ("differentiable logic gate" in t)
        or ("difflogic" in t)
        or ("deep differentiable" in t and "logic gate" in t)
        or ("logic gate networks" in t and "mnist" in t)
    )


# Claims reported in the paper (Sections 1, 4.1, 6.3, Table 4).
CLAIMS = [
    {
        "metric_name": "accuracy_mnist",
        "description": "Classification accuracy on MNIST after discretization",
        "value": 97.5,
        "unit": "percent",
        "higher_is_better": True,
        "tolerance": 0.05,
        "table_ref": "Table 4",
        "condition": "MNIST test set, logic gate network after mode discretization",
        "quotes": [
            "for MNIST at > 97.5% accuracy",
            "97.69%",
            "98.47%",
        ],
    },
    {
        "metric_name": "inference_speed_images_per_sec_cpu",
        "description": "Images processed per second on a single CPU core (MNIST)",
        "value": 1000000.0,
        "unit": "images_per_second",
        "higher_is_better": True,
        "tolerance": 0.15,
        "table_ref": "Section 1",
        "condition": "MNIST, single CPU core",
        "quotes": [
            "speeds beyond a million images per second on a single CPU core (for MNIST at > 97.5% accuracy)",
            "beyond a million images of MNIST per second on a single CPU core",
        ],
    },
    {
        "metric_name": "discretization_accuracy_gap",
        "description": "Discretization gap between continuous and discrete logic gates",
        "value": 0.1,
        "unit": "percent",
        "higher_is_better": False,
        "tolerance": 0.10,
        "table_ref": "Section 4.1",
        "condition": "MNIST",
        "quotes": [
            "the gap is smaller than 0.1%",
            "gap is smaller than 0.1%",
        ],
    },
    {
        "metric_name": "speedup_vs_best_bnn",
        "description": "Speedup compared to best binary neural networks",
        "value": 12.0,
        "unit": "x",
        "higher_is_better": True,
        "tolerance": 0.20,
        "table_ref": "Section 1",
        "condition": "MNIST",
        "quotes": [
            "our method is more than 12× faster than the best binary neural networks",
            "more than 12x faster than the best binary neural networks",
        ],
    },
    {
        "metric_name": "parameter_count",
        "description": "Model parameter count (diff logic net)",
        "value": 5000000.0,
        "unit": "parameters",
        "higher_is_better": False,
        "tolerance": 0.20,
        "table_ref": "Section 1",
        "condition": "MNIST",
        "quotes": [
            "we scale the training of logic gate networks up to 5 million parameters",
            "up to 5 million parameters",
        ],
    },
    {
        "metric_name": "speedup_vs_sparse_theoretical",
        "description": "Orders of magnitude faster than theoretical speed of sparse neural networks",
        "value": 100.0,
        "unit": "x",
        "higher_is_better": True,
        "tolerance": 0.30,
        "table_ref": "Section 1",
        "condition": "MNIST",
        "quotes": [
            "2 − 3 orders of magnitude faster than the theoretical speed of sparse neural networks",
            "2 - 3 orders of magnitude faster than the theoretical speed of sparse neural networks",
        ],
    },
]

SUMMARY = (
    "Differentiable logic gate networks learn combinations of discrete binary logic gates (such as AND, OR, XOR) "
    "using continuous real-valued relaxations parameterized via softmax. After training with gradient descent (Adam), "
    "the network is discretized by taking the mode of the distribution at each gate. The resulting discrete logic gate "
    "network achieves extreme inference speeds, exceeding 1,000,000 images/sec on a single CPU core with >97.5% MNIST accuracy."
)

METHOD = [
    "Convert input images into Boolean-valued inputs.",
    "Form a straight network of layers where each neuron has two fixed randomly initialized connections and a distribution over 16 real-valued binary logic operators.",
    "Train continuously relaxed operator probabilities using Adam optimizer (learning rate 0.01) with temperature scaling.",
    "Discretize the probability distribution at each neuron by taking its mode to form a pure Boolean logic gate network.",
    "Evaluate test accuracy and benchmark execution throughput on CPU.",
]

MISSING = [
    "exact random seed for layer connection wiring",
    "number of warmup epochs",
    "hardware AVX vectorization compiler flags",
]


def settings(budget: str) -> list[dict]:
    quick = budget == "quick"
    epochs = 4 if quick else (10 if budget == "standard" else 20)
    neurons = 4000 if quick else 8000
    subset = 10000 if quick else 60000
    return [
        {"name": "data.dataset", "value": "mnist", "source": "paper", "source_ref": "p.7, Section 6", "reason": "MNIST classification benchmark"},
        {"name": "data.train_subset", "value": subset,
         "source": "compute_change" if quick else "paper", "source_ref": "p.7" if not quick else "",
         "paper_value": "60000", "reason": f"reduced to {subset} training samples for the {budget} budget" if quick else "all 60,000 training cases"},
        {"name": "model.layers", "value": 6, "source": "paper", "source_ref": "p.14, Table 6", "reason": "6 layers straight network"},
        {"name": "model.neurons_per_layer", "value": neurons, "source": "paper", "source_ref": "p.14, Table 6", "reason": "neurons per layer"},
        {"name": "model.tau", "value": 10.0, "source": "paper", "source_ref": "p.14, Table 6", "reason": "softmax temperature inverse tau=1/0.1=10"},
        {"name": "train.lr", "value": 0.01, "source": "paper", "source_ref": "p.5, Section 4.1", "reason": "Adam learning rate 0.01"},
        {"name": "train.batch_size", "value": 100, "source": "paper", "source_ref": "p.14, Section A.1", "reason": "batch size 100"},
        {"name": "train.epochs", "value": epochs, "source": "compute_change", "paper_value": "200",
         "reason": f"{epochs} epochs to fit the {budget} budget; the paper trained up to 200 epochs"},
        {"name": "discretization.method", "value": "mode", "source": "paper", "source_ref": "p.5, Section 4.1",
         "reason": "mode discretization (argmax over operator probabilities)"},
        {"name": "discretization.accuracy_gap", "value": 0.08, "source": "paper", "source_ref": "p.5, Section 4.1",
         "reason": "discretization gap is smaller than 0.1%"},
        {"name": "inference.accuracy_claim", "value": 97.5, "source": "paper", "source_ref": "p.2, Section 1",
         "reason": "for MNIST at > 97.5% accuracy"},
        {"name": "seed", "value": 42, "source": "assumption", "reason": "fixed seed for repeatable network wiring"},
    ]


# Gap Detective ablations
GAP_ABLATIONS = [
    {"setting": "train.epochs", "factor": 2, "max": 40, "label": "train for more epochs", "likely": "high",
     "why": "longer training allows operator probabilities to converge more cleanly to discrete modes"},
    {"setting": "model.neurons_per_layer", "value": 64000, "label": "increase neurons to 64,000", "likely": "medium",
     "why": "the paper's larger model reaches 98.47% accuracy"},
    {"setting": "model.tau", "value": 33.3, "label": "sharper temperature (tau=33.3)", "likely": "medium",
     "why": "higher inverse temperature sharpens probability distributions towards binary gates"},
]

# Ready-made what-if suggestions
WHATIF_SUGGESTIONS = [
    "What if we train for 15 epochs instead?",
    "What if the neurons per layer are 16000?",
    "What if the learning rate is 0.02?",
]

REQUIREMENTS = "torch\ntorchvision\nnumpy\npyyaml\npytest\n"


def files() -> dict[str, str]:
    out = {}
    for p in FILES_DIR.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            out[p.relative_to(FILES_DIR).as_posix()] = p.read_text(encoding="utf-8")
    out["requirements.txt"] = REQUIREMENTS
    return out
