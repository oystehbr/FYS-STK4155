# Assignment 2

## Task 2.1 🏎🏎

### Prerequisites (None)

### Missing Functionality (None)

### Usage/ functionality
To run **./move.sh** -> we need to be inside the folder *assignment2*

#### To do 2.1 (moving all files):
**Two existing folders:** folder1, folder2. **One non-existing folder**: folder4

*Here are some examples:*

```bash
./move.sh folder1 folder2 
./move.sh folder1
./move.sh folder4 folder2
```
  1. Result: move every single file/ folder inside folder1 -> (to) folder2
  2. Result: errormessage, since the script needs at least two commmandline arguments
  3. Result: errormessage, since the "from"-directory do not exist, it will exit


#### To do 2.1a (specify the types of files):
**Two existing folders:** folder1, folder2

*Here are some examples:*

```bash
./move.sh folder1 folder2 .py 
./move.sh folder1 folder2 .py .txt
./move.sh folder1 folder2 F
```
  1. Result: move every single file in folder1 which includes ".py" and moves it to folder2
  2. Result: move every single file in folder1 which includes ".py" or ".txt" and moves it to folder2
  3. Result: move every single file in folder1 which includes "F" and moves it to folder2

#### To do 2.1b (If the dst-directory does not exist, user get the option to create it):
**One existing folders:** folder1. **One non-existing folder**: folder12

*Here are some examples:*

```bash
./move.sh folder1 folder12
./move.sh folder1 folder12 .py
```
  - Result in both cases: we will be asked to create the directory, since it do not exist. Will get the option (Y/N). Y: Yes and N: No


#### To do 2.1c (If the dst-directory does not exist, user can have the current date and time as the name of the directory):
**One existing folders:** folder1. **One non-existing folder**: folder12

*Here are some examples:*

```bash
./move.sh folder1 folder12
./move.sh folder1 folder12 .py
```
  - Result in both cases: we will be asked to create the directory, since it do not exist. Will get the option (Y/N). Y: Yes and N: No. (SAME as 2.1b). Further, you will get the option to have the name of the directory as todays date-time. Will get the same option here (Y/N).


## Task 2.2 ⏳✍🏼

### Prerequisites

The user need to add some text at the bottom of these files:

#### .zshrc (or .bashrc)
```bash
source ~/.bash_profile
source $PWD/path_to_assignment2/track.sh
```

#### .bash_profile
```bash
export LOGFILE=.local/share/logfile.log
```

### Missing Functionality (None)

### Functionality
The chunks i am presenting shows how you can the track tasks (using the track function) in a terminal ☺️
You will be provided with a different task tracker for different folders -> so you can have several tasks open in the different folders, but just one task at a time inside the same folder.

#### Start a task:
```bash
track start Clean the house
```
  - This will start a task with the label "Clean the house", if there are none ongoing task
  - If there are some ongoing task, then the user will be provided with an informative message

#### Stop a task:
```bash
track stop
```
  - This will stop an ongoing task (if there are some)
  - If there are none ongoing task, then the user will be provided with an informative message

#### Status of task:
```bash
track status
```
  - Will give information if there are some ongoing task or not

#### Log the tasks:
```bash
track log
```
  - Will log all the finished tasks
  - If there are some ongoing task, this will also be logged

#### Guide of task function:
```bash
track some_wrong_input
track star
track sto
```
  - The user will be provided with some guide of how to use the track-function

### Restrictions

Time of a task in "track log" will provide some wrong information if it last for over a month -> Assumption in my code: 1 month = 30 days

### Usage
Reopen the terminal after adding the prerequisites.
Then you can be in what ever folder you want and start using the track functionality.

