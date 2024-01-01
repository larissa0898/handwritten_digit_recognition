# Handwritten Digit Recognition

### Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Installation: 
To To get started, follow these steps:

1. Ensure you have Python installed. This project is compatible with Python Python 3.9.5 and above.

2. Clone the repository to your local machine and navigate to the project directory:
    ```bash
    git clone https://github.com/larissa0898/handwritten_digit_recognition.git
    cd repository
    ```

3. Create a virtual environment:
    ```bash
    python -m venv myenv
    ```

4. Activate your environment:
    On Windows:
    ```bash
    myenv\Scripts\activate
    ```
    
    On macOS and Linux:
    ```bash
    source myenv/bin/activate
    ```

5. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage: 

1. Run the code by executing the `main.py` script:
    ```bash
    python main.py
    ```
2. When prompted, make the following choices:

    - `Do you want to see some of the MNIST images?`:
        - Type `y` to view 6 MNIST images.
        - Type `n` to skip this step.

    - `Do you want to train a new model?`:
        - Type `y` to train a new model (This process takes approximately 8 minutes).
        - Type `n` to load a pre-existing model.

## References:
MNIST dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)