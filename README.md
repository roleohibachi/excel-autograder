# excel-autograder
A Gradescope autograder for excel documents.

GS "Programming" assignments accept files of any type, so there's no reason they have to be constrained to just executable code. This is an example of using python (with openpyxl) to grade student responses in an excel spreadsheet format.

The autograder looks for "key.xlsx" and compares it to the student submission. Each difference is weighted and scored. Per-item output is provided to the student. 

The key for my assignment is not provided here, because I don't want my students to have it! Email me if you would like a sample key. You should make your own answer key, then remove the answers and distribute it as a template. Excel's "protect sheet" and "locked cell" functionality is useful for ensuring students don't create differences between the key outside of their answer boxes. You will need to update the pandas "read_excel" commands to align to your key. 

Thanks to Joel and Chimi in Customer Support for answering some of my questions during development! Also thanks to our [previous instructor](https://github.com/czig) who wrote an excel autograder for Jupyter Notebook, which served as the starting point of this project.
