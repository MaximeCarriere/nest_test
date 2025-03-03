# Testing Framework for FelixNet

## Overview
This repository contains the testing framework for FelixNet, a neural simulation environment. The testing pipeline ensures the proper functionality of various network components and generates relevant graphical representations.


## Running Tests
To execute the complete testing suite, run:
```sh
python main.py
```
This will:
- Run all test cases for the FelixNet network.
- Generate plots based on test results.

## Test Components
### `test_runner.py`
Located in `testing_function/`, this script handles the execution of:
- **Auditory Network Tests**
- **Articulatory Network Tests**

### `config.py`
Contains configuration settings, including:
- Test modes (`auditory`, `articulatory`, `ca_size`, `ca_size_over_threshold`)
- Directory paths for outputs and network files

### `visualization.py`
Responsible for generating plots based on test results. It utilizes:
- `plot_tca()` for network activity visualization
- `plot_ca_size()` for cell assembly size analysis
- `plot_ca_size_thresh()` for cell assembly size analysis over threshold

## Output & Graphs
Generated plots are saved in:
```
./graph/
```
These include:
- `Audi_TCA_<pres>_presentations.png`
- `Arti_TCA_<pres>_presentations.png`
- `CA_Size_<pres>_presentations.png`
- `CA_Size_Threshold_<pres>_presentations.png`

## Contributing
If you'd like to contribute:
1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
For any issues or inquiries, please contact:
- **Your Name** (your.email@example.com)
- Open an issue in the repository

