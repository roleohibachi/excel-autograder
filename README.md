# excel-autograder
A Gradescope autograder for excel documents.

Our students do an engineering project that, by long custom, requires the use of a spreadsheet to calculate and submit answers. A couple years back, one of our more enterprising instructors wrote an autograder in standalone python, but sharing it was difficult because our university-provided computers don’t all play nice with jupyter notebook, which he used. I realized that Gradescope “programming assignments” don’t actually need to be programming-related at all – from the docs, “Students can submit … files of any file type”. That means that his python script, which uses pandas to read & grade Excel spreadsheets, could work!

I updated the filepaths, and changed the output format to JSON as explained in the documentation.

The end result is a GS assignment that allows students to submit their Excel workbook and get a grade immediately. 

This version has been sanitized of any academic content - you'll need to modify the grading logic to suit your assignment. I've included the "CFD tables" that account for partial credit, but they don't work unless you program them.
