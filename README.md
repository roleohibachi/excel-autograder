# excel-autograder
A Gradescope autograder for excel documents.

GS "Programming" assignments accept files of any type, so there's no reason they have to be constrained to just executable code. This is an example of using python (with openpyxl) to grade student responses in an excel spreadsheet format.

Thanks to Joel and Chimi in Customer Support for answering some of my questions during development!

Our students do an engineering project that, by long custom, requires the use of a spreadsheet to calculate and submit answers. A couple years back, one of our [more enterprising instructors](https://github.com/czig) wrote an autograder in standalone python, but sharing it was difficult because our university-provided computers don’t all play nice with jupyter notebook, which he used. 

I updated the filepaths, and changed the output format to JSON as explained in the GS documentation.

The end result is a GS assignment that allows students to submit their Excel workbook and get a grade (or a format error) immediately. 

This version has been sanitized of any useful academic content, so my students don't get hold of it. You'll need to modify the grading logic to suit your assignment. I've included the "CFD tables" that account for partial credit, but they won't work unless you program them.
