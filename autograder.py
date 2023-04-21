import sys
import numpy as np
import pandas as pd
import os
import json
import argparse

    
def main(args) -> int:

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl') #openpyxl doens't enforce excel data validation and warns you a LOT.

    sheet_name="Analysis"
    
    tests=list()

    def write_gradescope_output(tests):
        with open(os.path.abspath('../results/results.json'), 'w') as f:
            json.dump({
            "visibility": "visible",
            "stdout_visibility": "hidden",
            "tests": tests
        }, f)
        
    def fail(reason: str):
        write_gradescope_output([{
            "name": "Autograder Failure",
            "output": reason,
            "output_format": "md",
            "visibility": "visible",
            "status":"failed",
            "max_score": 50,
            "score":0
        }])

    def scale_column(df,col,scale):
        for (i,v) in df[col].items():
            df.at[i,col] *= scale
            #df.rename(columns={col:col.split('(')[0].strip()},inplace=True)
        return col.split('(')[0].strip()

    def grade(title,header,maxscore,numqs):

        def count_table_differences(header,file1=os.path.abspath(args.spreadsheet),file2=os.path.abspath(args.key),nrows=3,sheet_name=sheet_name):
            df1 = pd.read_excel(file1,sheet_name=sheet_name,header=header,nrows=nrows,index_col=0)
            df2 = pd.read_excel(file2,sheet_name=sheet_name,header=header,nrows=nrows,index_col=0)
            return (df1.round(1).compare(df2.round(1)).count().sum()/2,df1.round(1).eq(df2.round(1)))

        pt_per_q = maxscore/numqs
        (numwrong,results)=count_table_differences(header=header)

        return {
            "score": maxscore-numwrong*pt_per_q,
            "max_score": maxscore,
            "name": title,
            "output": results.replace(True,"\N{White Heavy Check Mark}").replace(False,"\N{Cross Mark}").to_markdown(),
            "output_format": "md",
            "tags": [],
            "visibility": "visible"
        }

    if not os.path.isfile(args.spreadsheet):
        fail("No spreadsheet found! Remember to submit BOTH your spreadsheet and your Tronview file.")
        return 0

    students = pd.read_excel(os.path.abspath(args.spreadsheet),sheet_name=sheet_name,header=1,nrows=4)
    documentation = pd.read_excel(os.path.abspath(args.spreadsheet),sheet_name=sheet_name).fillna('').iloc[6,1]
    
    if not documentation:
        fail("No documentation! I'm not allowed to grade that.")
        return 0

    #do tronview tasks
    trontest={
        "name": "TronView",
        "output_format": "md",
        "visibility": "visible"
    }

    if os.path.isfile(args.tronview):
        trontest.update({'output':"TronView file received. These are the flight paths you created in TronView. This format is used in the MDL: \n"})
        trontest.update({"status":"passed"})

        tronview = json.load(open(os.path.abspath(args.tronview)))

        routes=list()

        for linkedList in tronview['entities']['linkedLists']:
            if linkedList['category']=='ROUTE':
                routes.append(linkedList)

        for route in routes:
            trontest.update({'output':trontest['output']+route['name']+":\n"})
            routepoints = list()
            for point in tronview['entities']['steerpoints']:
                if 'category' in point and point['category']=='ROUTE_POINT':
                    if 'routeId' in point and point['routeId']==route['id']:
                        routepoints.append(point)

            def format_dd(dd: float,nsew_pos: str,nsew_neg: str) -> str:
                nsew = nsew_pos
                if dd < 0:
                    nsew = nsew_neg
                    dd = -dd
                mins,secs = divmod(dd*3600,60)
                degs,mins = divmod(mins,60)
                return str(int(degs)).zfill(2)+":"+str(int(mins)).zfill(2)+":"+str(round(secs,2)).zfill(5)+nsew

            for routepoint in routepoints:
                lat = format_dd(routepoint['position']['lat'],"N","S")
                lng = format_dd(routepoint['position']['lng'],"E","W")
                trontest.update({'output':trontest['output']+"\t"+lat+" "+lng+"\n"})

    else:
        trontest.update({"status":"failed"})
        trontest.update({"output": "No Tronview file received! Remeber to re-submit with both files once you complete it.\n\nDrawing range rings in TronView requires units of Nautical Miles (NM). Here are your Table 3 values, converted to NM: \n\n"})

        student_t3 = pd.read_excel(os.path.abspath(args.spreadsheet),sheet_name=sheet_name,header=22,nrows=3,index_col=0)
        for col in student_t3.columns.values:
            scale_column(student_t3,col=col,scale=0.54)
        trontest.update({'output':trontest['output']+student_t3.apply(np.ceil).astype(int).to_markdown()})

    tests.append(trontest)

    

    #grade tables
    power_range     = grade(title="Table 1: RADAR to Aircraft to RADAR Power-Range (km)", header=10, maxscore=10, numqs=5)
    los_range       = grade(title="Table 2: Site to Aircraft LOS Range (km)",             header=16, maxscore=10, numqs=5)
    detection_range = grade(title="Table 3: Site's Detection Range of Aircraft (km)",     header=22, maxscore=10, numqs=5)
    burnthru_range  = grade(title="Table 4: RADAR Burn-Through Power-Range (km)",         header=28, maxscore=10, numqs=3)
    rwr_range       = grade(title="Table 5: Aircraft RWR Power-Range (km)",header=34, maxscore=10, numqs=5)
    tests.extend([power_range,los_range,detection_range,burnthru_range,rwr_range])
    
    write_gradescope_output(tests)
    return 0

if __name__ == '__main__':

    #minimize traceback
    sys.tracebacklimit=1

    parser = argparse.ArgumentParser()
    parser.add_argument( '-s','--spreadsheet', default="./submission.xlsx", help='Submission spreadsheet')
    parser.add_argument( '-k','--key', default="./key.xlsx", help='Key spreadsheet')
    parser.add_argument( '-t','--tronview', default="./tronview.txt", help='Tronview file (optional)', required=False)
    args=parser.parse_args()
    sys.exit(main(args))

