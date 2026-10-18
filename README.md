# Tugas Besar 1 IF3170 Inteligensi Artifisial: Local Search

Kelompok G59: LazyService

Buka laporan lengkap disini: https://docs.google.com/document/d/1VKHKJmA1G6BkhgAVuTR5mnTHwH1mBS7mWWuiv9UqZRQ/edit?tab=t.riwja32n1khd

Program untuk mencari konfigurasi pengisian paket ke dalam peti kemas truk dengan memaksimalkan total _value_ paket yang diangkut, menggunakan algoritma _local search_. Setiap state harus memenuhi constraint berikut:

- paket tidak boleh menembus dinding peti kemas atau saling bertabrakan,
- paket tidak boleh melayang (minimal satu titik alasnya harus ditopang lantai atau paket lain),
- paket _fragile_ tidak boleh menjadi penopang paket lain,
- total berat tidak boleh melebihi kapasitas truk.

Algoritma yang diimplementasikan:

- Hill climbing
  - steepest ascent
  - with sideways move
  - random restart
  - stochastic
- Simulated annealing
- Genetic algorithm

Mendukung lebih dari satu truk (bonus multi-truk). Jumlah truk ditentukan di file input.

## Struktur Repository

```
.
├── docs/                   Laporan
├── input/                  File input JSON
│   ├── satu_truk.json
│   └── multi_truk.json
├── output/                 Hasil run
├── src/
│   ├── main.py             Entry point
│   ├── core/               Model data, state, constraint, input parser, upper bound
│   ├── algorithms/         Hill-Climbing, Simulated Annealing, Genetic Algorithm
│   ├── experiments/        Parameter, runner eksperimen, penyimpanan output
│   ├── tui/                Menu dan tampilan terminal
│   └── visualization/      Visualisasi 3D state dan plot hasil
└── requirements.txt
```

## Setup

Prerequisite: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: `matplotlib`, `numpy`.

## Cara Menjalankan

Jalankan dari root repository dengan file input JSON sebagai argumen:

```bash
python src/main.py input/satu_truk.json
python src/main.py input/multi_truk.json --seed 10
```

| Argumen  | Keterangan                                              |
| -------- | ------------------------------------------------------- |
| `input`  | **Required.** Path file JSON berisi data truk dan paket |
| `--seed` | Optional. Random seed (default `67`)                    |

## Format Input

File input berupa JSON dengan daftar truk dan paket. Contoh:

```json
{
  "trucks": [
    {
      "id": "Truk-1",
      "dimensions": { "w": 6, "l": 6, "h": 5 },
      "max_capacity": 70
    }
  ],
  "packages": [
    {
      "id": "P1",
      "dimensions": { "w": 3, "l": 2, "h": 2 },
      "value": 35,
      "weight": 11,
      "is_fragile": false,
      "eta": 2
    },
    {
      "id": "P2",
      "dimensions": { "w": 2, "l": 2, "h": 2 },
      "value": 25,
      "weight": 8,
      "is_fragile": false,
      "eta": 3
    }
  ]
}
```

Minimal harus terdapat 1 truk dan 2 paket.
