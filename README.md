## Clarion integrated DIVA model
​
The DIVA model integrated into Clarion using the pyClarion library. Tested on the 6 elemental categories created from inputs of three binary features represented as values of ±1 for each dimension. Parameters are set to the following:
* standard deviation: 0.5
* learning rate: 0.1
* degree of focus ($\beta$): 10
* activation function: Tanh
* optimization function: Adam
​
 ## Development requirements
  * Programming Languages: Python
  * Software Dependencies: python 3.12, pip
  * Librarys: [Pyclarion branch v2409](https://github.com/cmekik/pyClarion/tree/v2409)
    
 ## Installation and Instructions
 In the terminal, run `pip install -r requirements.txt`  
 Then follow the instructions on the pyClarion repository to install pyClarion. After, simply run main.py using the command `python main.py` in the terminal. This will create a new set of graphs within the graphs and focus_graphs repositories as well as new averages.cvc and focus_averages.cvc documenting the Mean error rate for each epoch. 
 
