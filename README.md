# Supplementary Material: Towards a Parsimonious Computational Marxian Model

## Authors
- Sandy H. S. Herho
- Siti N. Kaban

## Abstract
This repository contains the computational implementation, data, and supplementary materials for our research on a parsimonious Marxian economic model. The project explores worker-capital dynamics through a minimal yet robust computational approach.

## Repository Structure
```
.
├── data/
│   └── base_case.csv
├── figs/
│   ├── figure1.png
│   ├── figure2.png
│   ├── figure3.png
│   └── figure4.png
├── marxian_model.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Dependencies
- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Pandas (optional, for data export)

## Installation
```bash
git clone https://github.com/username/parsimonious-marxian-model.git
cd parsimonious-marxian-model
pip install -r requirements.txt
```

## Reproducing Results
### Run Simulation
```bash
python marxian_model.py
```

## Figures Description
1. **Figure 1** (`figure1.png`):
   - Phase portrait visualization
   - System dynamics in worker's share-capital space

2. **Figure 2** (`figure2.png`):
   - Time series of worker's share, capital, and profit rates

3. **Figure 3** (`figure3.png`):
   - Parameter variation across different class struggle intensities

4. **Figure 4** (`figure4.png`):
   - Crisis scenario comparisons

## Data
- `base_case.csv`: Simulation data for default model parameters

## Citation
If you use this model or code in your research, please cite as:

```bibtex
@article{herho2025parsimonious,
  title = {Towards a Parsimonious Computational Marxian Model},
  author = {Herho, Sandy H. S. and Kaban, Siti N.},
  year = {2025},
  journal = {[Journal Name]},
  volume = {[Volume]},
  number = {[Issue]},
  pages = {[Pages]}
}
```

## License
This project is licensed under the [WTFPL](LICENSE) - Do What The Fuck You Want To Public License.

## Contact
[Sandy H. S. Herho](mailto:sandy.herho@email.ucr.edu)
