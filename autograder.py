import pandas as pd
import numpy as np
import math
import os
import re
import json

#minimize traceback for sanity (stdout should be hidden from students anyway)
import sys
sys.tracebacklimit=1

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl') #openpyxl doens't enforce excel data validation and warns you a LOT.

proposals_filename = os.path.abspath("./submission.xlsx")
filepath = os.path.abspath("./submission.xlsx")

class project2_grader:
    def __init__(self, table2, table3_weights, table3, table4, design_parameters_table, requirements_table):

        #In our project, we have a table of "given" values
        self.requirements=requirements_table.T.to_dict()
        
        #define input dataframe
        self.design_parameters_df = design_parameters_table
        
        #define submission dataframes
        self.table2_df = table2
        self.table3_weights_df = table3_weights
        self.table3_df = table3
        self.table4_df = table4
        self.stud_cost_weight = self.str_to_float(self.table3_weights_df.loc[0,'Cost Weight'])
        self.stud_mtbf_weight = self.str_to_float(self.table3_weights_df.loc[0,'MTBF Weight'])
        
        #define solution dataframes
        self.table2_soln = pd.DataFrame().reindex_like(self.table2_df)
        self.table2_soln['Design'] = self.table2_df['Design'].copy(deep=True)
        self.table3_soln = pd.DataFrame().reindex_like(self.table3_df)
        self.table3_soln['Design'] = self.table3_df['Design'].copy(deep=True)
        self.table4_soln = pd.DataFrame().reindex_like(self.table4_df)
        
        #define "correct for data" tables - this allows for partial credit based on previous incorrect answers
        self.table2_cfd = pd.DataFrame().reindex_like(self.table2_df)
        self.table2_cfd['Design'] = self.table2_df['Design'].copy(deep=True)
        self.table3_cfd = pd.DataFrame().reindex_like(self.table3_df)
        self.table3_cfd['Design'] = self.table3_df['Design'].copy(deep=True)
        self.table4_cfd = pd.DataFrame().reindex_like(self.table4_df)
                
        #defaults
        self.grade = 75
        self.eng_analysis_grade = 50
        self.decision_matrix_grade = 25
        self.grade_notes = []
        self.graded = False
        
        #create solution
        self.gen_table4_solution()
        self.gen_table4_cfd()
        
    def calc_answer1(self, vmax, vmin, bits):
        return 3 #build your grading logic here
                              
    def calc_answer2(self, sample_rate, bits_per_sample, bytes_per_unit):
        return 3 #build your grading logic here                              
        
    def calc_answer3(self, sample_rate, bits_per_sample, upload_rate):
        return 3 #build your grading logic here    
                              
    def calc_answer4(self, unit_cost):
        return unit_cost*self.requirements['Total Units']['Value']
        
    def is_within_tolerance(self, studVal, solnVal, tolerance):
        #studVal is a number, solnVal is a number, tolerance is a decimal between 0 and 1
        if solnVal >= 0:
            #keep as you would expect for numbers 0 or greater
            if (1-tolerance)*solnVal <= studVal <= (1+tolerance)*solnVal:
                return True
            else:
                return False
        else:
            #flip when correct answer is negative
            if (1+tolerance)*solnVal <= studVal <= (1-tolerance)*solnVal:
                return True
            else:
                return False            
        
    def str_to_float(self, string_val):
        if type(string_val) == str:
            # check for digits within a string
            result = re.search(r'[-+]?\d+\.?\d+',string_val)
            if result:
                #pull digits out of string and convert to float
                return float(result[0])
            else:
                #no digits, keep string as is
                return string_val
        else:
            # check if a numpy float
            if type(string_val) == np.float64:
                # check if numpy float is NaN
                if math.isnan(string_val):
                    # if NaN, then student left blank. Return -1 because negative number
                    # not likely to be a correct answer 
                    return -1
                else:
                    return string_val
            else:
                return string_val
            
    def calc_answer5(self, v_in, v_min, resolution, bits_per_sample):
        return "the answer"
                              
    def gen_table2_solution(self):
        for i,row in self.design_parameters_df.iterrows():
            #get values -- only a convenience step
            unit_cost = self.design_parameters_df.loc[i,'Unit Cost']
            vmax = self.design_parameters_df.loc[i,'Voltage Max']
            vmin = self.design_parameters_df.loc[i,'Voltage Min']
            bits_per_sample = self.design_parameters_df.loc[i,'Bits per Sample']
            sample_rate = self.design_parameters_df.loc[i,'Sampling Freq']
            bytes_per_unit = self.design_parameters_df.loc[i,'Size']
            upload_rate = self.design_parameters_df.loc[i,'Upload Data Rate']
            mtbf = self.design_parameters_df.loc[i,'MTBF']
            
            #do calculations and store in solution table
            self.table2_soln.loc[i,'ADC Resolution'] = self.calc_answer1(vmax, vmin, bits_per_sample)
            self.table2_soln.loc[i,'Storage Duration'] = self.calc_answer2(sample_rate, \
                                                                                      bits_per_sample, bytes_per_unit)
            self.table2_soln.loc[i,'Duration to Upload'] = self.calc_answer3(sample_rate, \
                                                                                          bits_per_sample, upload_rate)
            self.table2_soln.loc[i,'Total System Cost'] = self.calc_answer4(unit_cost)
            
            #check requirements
            if self.table2_soln.loc[i,'ADC Resolution'] > self.requirements['ADC resolution']['Value'] or \
               self.table2_soln.loc[i,'Storage Duration'] < self.requirements['Storage Duration']['Value'] or \
               self.table2_soln.loc[i,'Duration to Upload'] > self.requirements['Duration to Upload']['Value']:
                self.table2_soln.loc[i,'Adjudication'] = 'FAIL'
            else:
                self.table2_soln.loc[i,'Adjudication'] = 'PASS'
                
    def gen_table3_solution(self):
        #need to have table 2 before table 3
        self.gen_table2_solution()
        
        #fill in raw values from table 2
        for i,row in self.table2_cfd.iterrows():
            self.table3_soln.loc[i,'Raw'] = self.table2_soln.loc[i,'Total System Cost']
            self.table3_soln.loc[i,'Raw.1'] = self.design_parameters_df.loc[i,'MTBF']  
        
        #delete the failures 
        failures = self.table2_soln[self.table2_soln['Adjudication'].str.lower() == 'fail'].index
        self.table3_soln = self.table3_soln.drop(index=failures).reset_index(drop=True)          
                
        #get min and max for each row
        min_cost = self.table3_soln['Raw'].min()
        max_mtbf = self.table3_soln['Raw.1'].max()
        
        #normalized, weighted, and total values for each proposal 
        self.table3_soln['Norm'] = self.table3_soln.apply(lambda row: min_cost/row['Raw'], axis=1)
        self.table3_soln['Norm.1'] = self.table3_soln.apply(lambda row: row['Raw.1']/max_mtbf, axis=1)
        self.table3_soln['Weighted'] = self.table3_soln.apply(lambda row: row['Norm']*self.requirements['Cost Weight']['Value'], axis=1)  
        self.table3_soln['Weighted.1'] = self.table3_soln.apply(lambda row: row['Norm.1']*self.requirements['MTBF weight']['Value'], axis=1)
        self.table3_soln['Total'] = self.table3_soln['Weighted'] + self.table3_soln['Weighted.1']
        
        #find winner
        max_total = self.table3_soln['Total'].max()
        self.table3_soln['Award Contract'] = self.table3_soln.apply(lambda row: \
                                                                    'YES' if row['Total'] == max_total else 'NO', \
                                                                    axis=1)
        
    def gen_table4_solution(self):
        #need to have table 3 before table 4
        self.gen_table3_solution()
        
        #find which proposal gets contract
        # This is very complicated because the index given from table 3 is not the real index
        # that is listed on the design parameters, so we have to use the index to select the 'Design'
        # column, then pull the number out of that text with a regex, then we can take that number minus 1
        # as our real index
        awarded = self.table3_soln[self.table3_soln['Award Contract'].str.lower() == 'yes']
        awarded_design = awarded.loc[awarded.index[0],'Design']
        proposal_num = re.search(r'\d',awarded_design)[0]
        proposal_idx = int(proposal_num)-1

        #givens
        in_max = 0.080
        in_min = -0.025
        R = 10000
        v_input_transducer = -0.005
        vmax = self.design_parameters_df.loc[proposal_idx,'Voltage Max']
        vmin = self.design_parameters_df.loc[proposal_idx,'Voltage Min']
        sample_rate = self.design_parameters_df.loc[proposal_idx,'Sampling Freq']   
        bits_per_sample = self.design_parameters_df.loc[proposal_idx,'Bits per Sample'] 
        
        #pull resolution from table2
        resolution = self.table2_soln.loc[proposal_idx,'ADC Resolution']*1e-6 #Resolutions are in uV
        
        #calculations
        self.table4_soln.loc[0,'Gain'] = (vmax-vmin)/(in_max-in_min)
        self.table4_soln.loc[0,'Bias'] = (vmax - self.table4_soln.loc[0,'Gain']*in_max)*1000 #must be mV
        self.table4_soln.loc[0,'Filter Type'] = 'LPF'
        self.table4_soln.loc[0,'fc/o'] = (sample_rate/2)/1000 #must be in kHz
        self.table4_soln.loc[0,'Capacitance'] = (1/(2*math.pi*R*self.table4_soln.loc[0,'fc/o']*1e3))*1e9 #must be nF
        v_in = v_input_transducer*self.table4_soln.loc[0,'Gain'] + (self.table4_soln.loc[0,'Bias']/1000) 
        self.table4_soln.loc[0,'Binary Output'] = self.calc_answer5(v_in, vmin, resolution, bits_per_sample)
        
    def gen_table2_cfd(self):
        for i,row in self.design_parameters_df.iterrows():
            #get values -- only a convenience step
            unit_cost = self.design_parameters_df.loc[i,'Unit Cost']
            vmax = self.design_parameters_df.loc[i,'Voltage Max']
            vmin = self.design_parameters_df.loc[i,'Voltage Min']
            bits_per_sample = self.design_parameters_df.loc[i,'Bits per Sample']
            sample_rate = self.design_parameters_df.loc[i,'Sampling Freq']
            bytes_per_unit = self.design_parameters_df.loc[i,'Size']
            upload_rate = self.design_parameters_df.loc[i,'Upload Data Rate']
            mtbf = self.design_parameters_df.loc[i,'MTBF']
            
            #do calculations and store in solution table
            self.table2_cfd.loc[i,'ADC Resolution'] = self.calc_answer1(vmax, vmin, bits_per_sample)
            self.table2_cfd.loc[i,'Storage Duration'] = self.calc_answer2(sample_rate, \
                                                                                      bits_per_sample, bytes_per_unit)
            self.table2_cfd.loc[i,'Duration to Upload'] = self.calc_answer3(sample_rate, \
                                                                                          bits_per_sample, upload_rate)
            self.table2_cfd.loc[i,'Total System Cost'] = self.calc_answer4(unit_cost)
            
            #check requirements (CFD here based on their calcs)
            stud_adc_resolution = self.str_to_float(self.table2_df.loc[i,'ADC Resolution'])
            stud_storage_duration = self.str_to_float(self.table2_df.loc[i,'Storage Duration'])
            stud_duration_to_upload = self.str_to_float(self.table2_df.loc[i,'Duration to Upload'])
            if stud_adc_resolution > self.requirements['ADC resolution']['Value'] or \
               stud_storage_duration < self.requirements['Storage Duration']['Value'] or \
               stud_duration_to_upload > self.requirements['Duration to Upload']['Value']:
                self.table2_cfd.loc[i,'Adjudication'] = 'FAIL'
            else:
                self.table2_cfd.loc[i,'Adjudication'] = 'PASS'
                
    def gen_table3_cfd(self):
        #need to have table 2 before table 3
        self.gen_table2_cfd()
        
        #fill in raw values from student's table 2 
        for i,row in self.table2_cfd.iterrows():
            self.table3_cfd.loc[i,'Raw'] = self.str_to_float(self.table2_df.loc[i,'Total System Cost'])
            self.table3_cfd.loc[i,'Raw.1'] = self.design_parameters_df.loc[i,'MTBF']
        
        #delete the failures based on student's table 2
        failures = self.table2_df[self.table2_df['Adjudication'].str.lower() == 'fail'].index
        self.table3_cfd = self.table3_cfd.drop(index=failures).reset_index(drop=True) 
        self.table3_df = self.table3_df.drop(index=failures).reset_index(drop=True)
                
        #get min and max for each raw value column of student's table 3
        min_cost = self.table3_df['Raw'].min()
        max_mtbf = self.table3_df['Raw.1'].max()
        
        #normalized, weighted, and total values for each proposal  (CFD allowed for any of these
        #so use submission values)
        self.table3_cfd['Norm'] = self.table3_df.apply(lambda row: min_cost/row['Raw'], axis=1)
        self.table3_cfd['Norm.1'] = self.table3_df.apply(lambda row: row['Raw.1']/max_mtbf, axis=1)
        self.table3_cfd['Weighted'] = self.table3_df.apply(lambda row: row['Norm']*self.stud_cost_weight, axis=1)
        self.table3_cfd['Weighted.1'] = self.table3_df.apply(lambda row: row['Norm.1']*self.stud_mtbf_weight, axis=1)
        self.table3_cfd['Total'] = self.table3_df['Weighted'] + self.table3_df['Weighted.1']
        
        #find winner
        max_total = self.table3_cfd['Total'].max()
        self.table3_cfd['Award Contract'] = self.table3_cfd.apply(lambda row: \
                                                                    'YES' if row['Total'] == max_total else 'NO', \
                                                                    axis=1)
        
    def gen_table4_cfd(self):
        #need to have table 3 built before this can be run
        self.gen_table3_cfd()

        #find which proposal gets contract
        # This is very complicated because the index given from table 3 is not the real index
        # that is listed on the design parameters, so we have to use the index to select the 'Design'
        # column, then pull the number out of that text with a regex, then we can take that number minus 1
        # as our real index
        awarded = self.table3_df[self.table3_df['Award Contract'].str.lower() == 'yes']
        awarded_design = awarded.loc[awarded.index[0],'Design']
        proposal_num = re.search(r'\d',awarded_design)[0]
        proposal_idx = int(proposal_num)-1    
        
        #givens
        in_max = 0.080
        in_min = -0.025
        R = 10000
        v_input_transducer = -0.005
        vmax = self.design_parameters_df.loc[proposal_idx,'Voltage Max']
        vmin = self.design_parameters_df.loc[proposal_idx,'Voltage Min']
        sample_rate = self.design_parameters_df.loc[proposal_idx,'Sampling Freq']  
        bits_per_sample = self.design_parameters_df.loc[proposal_idx,'Bits per Sample'] 
        
        #pull resolution from table2
        resolution = self.table2_df.loc[proposal_idx,'ADC Resolution']*1e-6 #resolutions are in uV
        
        #calculations
        self.table4_cfd.loc[0,'Gain'] = (vmax-vmin)/(in_max-in_min)
        self.table4_cfd.loc[0,'Bias'] = (vmax - self.table4_df.loc[0,'Gain']*in_max)*1000 #must be mV
        self.table4_cfd.loc[0,'Filter Type'] = 'LPF'
        self.table4_cfd.loc[0,'fc/o'] = (sample_rate/2)/1000 # must be kHz
        self.table4_cfd.loc[0,'Capacitance'] = (1/(2*math.pi*R*self.table4_df.loc[0,'fc/o']*1e3))*1e9 #must be nF
        v_in = v_input_transducer*self.table4_df.loc[0,'Gain'] + (self.table4_df.loc[0,'Bias']/1000) 
        self.table4_cfd.loc[0,'Binary Output'] = self.calc_answer5(v_in, vmin, resolution, bits_per_sample)
        
    def calc_grade(self,late_flag):
        #table 2
        #no CFD: ADC resolution, Total Message Power, Storage Duration, Duration to Upload, Total System Cost
        #CFD: Single Sideband Voltage, Modulation Index (alpha), Modulation Type, Modulation Efficiency (as %), and
        #     Adjudication
        for i, row in self.table2_soln.iterrows():
            prop_num = i + 1
            #resolution (round to nearest integer)
            stud_resolution = round(self.str_to_float(self.table2_df.loc[i,'ADC Resolution']))
            soln_resolution = round(self.table2_soln.loc[i,'ADC Resolution'])
            if not(self.is_within_tolerance(stud_resolution, soln_resolution, 0.005)):
                self.grade -= 2
                self.eng_analysis_grade -= 2
                self.grade_notes.append('Incorrect ADC resolution for Proposal %d (-2pts)' % prop_num)
            #Storage duration (round to 1 decimal place)
            stud_storage_duration = round(self.str_to_float(self.table2_df.loc[i,'Storage Duration']),1)
            soln_storage_duration = round(self.table2_soln.loc[i,'Storage Duration'],1)
            if not(self.is_within_tolerance(stud_storage_duration, soln_storage_duration, 0.01)):
                self.grade -= 2
                self.eng_analysis_grade -= 2
                self.grade_notes.append('Incorrect Storage Duration for Proposal %d (-2pts)' % prop_num)
            #Duration for upload (round to 1 decimal place)
            stud_upload_duration = round(self.str_to_float(self.table2_df.loc[i,'Duration to Upload']),1)
            soln_upload_duration = round(self.table2_soln.loc[i,'Duration to Upload'],1)
            if not(self.is_within_tolerance(stud_upload_duration, soln_upload_duration, 0.01)):
                self.grade -= 2
                self.eng_analysis_grade -= 2
                self.grade_notes.append('Incorrect Duration to Upload for Proposal %d (-2pts)' % prop_num) 
            #System cost (round to nearest integer)
            stud_cost = round(self.str_to_float(self.table2_df.loc[i,'Total System Cost']))
            soln_cost = round(self.table2_soln.loc[i,'Total System Cost'])
            if not(self.is_within_tolerance(stud_cost, soln_cost, 0.00001)):
                self.grade -= 2
                self.eng_analysis_grade -= 2
                self.grade_notes.append('Incorrect Total System Cost for Proposal %d (-2pts)' % prop_num)
            #Adjudication (text)
            stud_adjud = str(self.table2_df.loc[i,'Adjudication']).lower()
            cfd_adjud = self.table2_cfd.loc[i,'Adjudication'].lower()
            soln_adjud = self.table2_soln.loc[i,'Adjudication'].lower()
            if stud_adjud == soln_adjud:
                pass
            elif stud_adjud == cfd_adjud:
                self.grade_notes.append('CFD Adjudication for Proposal %d' % prop_num)
            else:
                self.grade -= 3
                self.eng_analysis_grade -= 3
                self.grade_notes.append('Incorrect Adjudication for Proposal %d (-3pts)' % prop_num)    
                
        #table 3 weights
        if self.stud_cost_weight != 0.75:
            self.grade -= 1
            self.decision_matrix_grade -= 1
            self.grade_notes.append('Incorrect cost weight (-1pt)')
        if self.stud_mtbf_weight != 0.25:
            self.grade -= 1
            self.decision_matrix_grade -= 1
            self.grade_notes.append('Incorrect MTBF weight (-1pt)')
        
        #table 3
        #no CFD: none
        #CFD: raw (for matching table 2 values), norm, weighted, total, awarded contract
        for i,row in self.table3_cfd.iterrows():
            proposal = self.table3_df.loc[i,'Design']
            #cost raw value - round to nearest integer (can lose points here if they don't use their value from table 2)
            stud_table3_cost = round(self.str_to_float(self.table3_df.loc[i,'Raw']))
            cfd_table3_cost = round(self.table3_cfd.loc[i,'Raw'])
            if self.is_within_tolerance(stud_table3_cost, cfd_table3_cost, 0.00001):
                pass
            else:
                self.grade -= 1
                self.decision_matrix_grade -= 1
                self.grade_notes.append('Incorrect cost raw value for %s (does not match table 2) (-1pt)' \
                                        % proposal)
            #cost normalized value (round to three decimal places)
            stud_cost_norm = round(self.table3_df.loc[i,'Norm'],3)
            cfd_cost_norm = round(self.table3_cfd.loc[i,'Norm'],3)
            if self.is_within_tolerance(stud_cost_norm, cfd_cost_norm, 0.01):
                pass
            else:
                self.grade -= 1
                self.decision_matrix_grade -= 1
                self.grade_notes.append('Incorrect Cost norm value for %s (-1pt)' % proposal)
            #cost weighted value (round to three decimal places)
            stud_cost_weight = round(self.table3_df.loc[i,'Weighted'],3)
            cfd_cost_weight = round(self.table3_cfd.loc[i,'Weighted'],3)
            if self.is_within_tolerance(stud_cost_weight, cfd_cost_weight, 0.01):
                pass
            else:
                self.grade -= 1
                self.decision_matrix_grade -= 1
                self.grade_notes.append('Incorrect Cost weighted value for %s (-1pt)' % proposal)
            #MTBF norm value (round to three decimal places)
            stud_mtbf_norm = round(self.table3_df.loc[i,'Norm.1'],3)
            cfd_mtbf_norm = round(self.table3_cfd.loc[i,'Norm.1'],3)
            if self.is_within_tolerance(stud_mtbf_norm, cfd_mtbf_norm, 0.02):
                pass
            else:
                self.grade -= 1
                self.decision_matrix_grade -= 1
                self.grade_notes.append('Incorrect MTBF norm value for %s (-1pt)' % proposal)  
            #MTBF weighted value (round to three decimal places)
            stud_mtbf_weight = round(self.table3_df.loc[i,'Weighted.1'],3)
            cfd_mtbf_weight = round(self.table3_cfd.loc[i,'Weighted.1'],3)
            if self.is_within_tolerance(stud_mtbf_weight, cfd_mtbf_weight, 0.02):
                pass
            else:
                self.grade -= 1
                self.decision_matrix_grade -= 1
                self.grade_notes.append('Incorrect MTBF weighted value for %s (-1pt)' % proposal) 
            #total weighted value (round to three decimal places)
            stud_total = round(self.table3_df.loc[i,'Total'],3)
            cfd_total = round(self.table3_cfd.loc[i,'Total'],3)
            if self.is_within_tolerance(stud_total, cfd_total, 0.005):
                pass
            else:
                self.grade -= 1
                self.decision_matrix_grade -= 1
                self.grade_notes.append('Incorrect Total weighted value for %s (-1pt)' % proposal)
            #awarded
            stud_awarded = str(self.table3_df.loc[i,'Award Contract']).lower()
            cfd_awarded = self.table3_cfd.loc[i,'Award Contract'].lower()
            if stud_awarded == cfd_awarded:
                pass
            else:
                self.grade -=2
                self.decision_matrix_grade -= 2
                self.grade_notes.append('Incorrect "Award Contract" for %s (-2pts)' % proposal)
                
        #table 4
        #always giving CFD for awarded
        #no extra CFD: Gain, filter type, cutoff freq
        #extra CFD: bias, capacitance
        #gain - (round to 3 decimal places)
        stud_table4_gain = round(self.str_to_float(self.table4_df.loc[0,'Gain']),3)
        cfd_table4_gain = round(self.table4_cfd.loc[0,'Gain'],3)
        if self.is_within_tolerance(stud_table4_gain, cfd_table4_gain, 0.01):
            pass
        else:
            self.grade -= 2
            self.eng_analysis_grade -= 2
            self.grade_notes.append('Incorrect gain in table 4 (-2pts)')     
        #bias - (round to 3 decimal places)
        stud_table4_bias = round(self.str_to_float(self.table4_df.loc[0,'Bias']),3)
        cfd_table4_bias = round(self.table4_cfd.loc[0,'Bias'],3)
        if self.is_within_tolerance(stud_table4_bias, cfd_table4_bias, 0.01):
            pass
        else:
            self.grade -= 2
            self.eng_analysis_grade -= 2
            self.grade_notes.append('Incorrect bias in table 4 (-2pts)')   
        #filter type - string
        stud_table4_filt = str(self.table4_df.loc[0,'Filter Type']).lower()
        cfd_table4_filt = self.table4_cfd.loc[0,'Filter Type'].lower()
        if stud_table4_filt == cfd_table4_filt:
            pass
        else:
            self.grade -= 2
            self.eng_analysis_grade -= 2
            self.grade_notes.append('Incorrect filter type in table 4 (-2pts)')   
        #cutoff freq - (round to one decimal place)
        stud_table4_cutoff = round(self.str_to_float(self.table4_df.loc[0,'fc/o']),1)
        cfd_table4_cutoff = round(self.table4_cfd.loc[0,'fc/o'],1)
        if self.is_within_tolerance(stud_table4_cutoff, cfd_table4_cutoff, 0.005):
            pass
        else:
            self.grade -= 2
            self.eng_analysis_grade -= 2
            self.grade_notes.append('Incorrect cutoff frequency in table 4 (-2pts)')   
        #cutoff freq - (round to 11 decimal places)
        stud_table4_cap = round(self.str_to_float(self.table4_df.loc[0,'Capacitance']),11)
        cfd_table4_cap = round(self.table4_cfd.loc[0,'Capacitance'],11)
        if self.is_within_tolerance(stud_table4_cap, cfd_table4_cap, 0.01):
            pass
        else:
            self.grade -= 2
            self.eng_analysis_grade -= 2
            self.grade_notes.append('Incorrect capacitance in table 4 (-2pts)')  
        #binary output - must be within 1% tolerance of decimal number and binary string must 
        #be correct length
        stud_table4_bin = self.table4_df.loc[0,'Binary Output']
        cfd_table4_bin = self.table4_cfd.loc[0,'Binary Output']
        stud_table4_dec = int(stud_table4_bin,2)
        cfd_table4_dec = int(cfd_table4_bin,2)
        if self.is_within_tolerance(stud_table4_dec, cfd_table4_dec, 0.01) and \
        len(stud_table4_bin) == len(cfd_table4_bin):
            pass
        else:
            self.grade -= 2
            self.eng_analysis_grade -= 2
            self.grade_notes.append('Incorrect binary output in table 4 (-2pts)')  
            #print('student binary, cfd binary: ', stud_table4_bin, cfd_table4_bin)
            #print('student decimal, cfd decimal: ', stud_table4_dec, cfd_table4_dec)
            #print('student bin len, cfd bin len: ',len(stud_table4_bin),len(cfd_table4_bin))
        
            
        #apply late penalty if late flag is true
        if late:
            late_penalty = math.floor(75*0.25)
            self.grade -= late_penalty
            self.grade_notes.append('Late penalty of -25% off available points (-' + str(late_penalty) + 'pts)')
        #set graded flag to true so we know we can trust graded values    
        self.graded = True 
        return (self.eng_analysis_grade, self.decision_matrix_grade, self.grade_notes)

#pull in proposal information and clean it so that it is all in same units
df_proposals = pd.read_excel(proposals_filename, sheet_name = "Table 1 Design Proposals", header=3, \
                             nrows=5)
#cast value columns as float
value_cols = ['Unit Cost','Voltage Max','Voltage Min','Sampling Freq','Bits per Sample', \
              'Size', 'Upload Data Rate', 'MTBF']
for col in value_cols:
    df_proposals[col] = df_proposals[col].astype(float)

#convert to standard units
for i, row in df_proposals.iterrows():
    if row['Unnamed: 3'] == 'mV':
        df_proposals.at[i,'Voltage Max'] *= 1e-3
    if row['Unnamed: 5'] == 'mV':
        df_proposals.at[i,'Voltage Min'] *= 1e-3
    if row['Unnamed: 7'] == 'kHz':
        df_proposals.at[i,'Sampling Freq'] *= 1e3
    if row['Unnamed: 12'] == 'Mbit/s':
        df_proposals.at[i,'Upload Data Rate'] *= 1e6
    elif row['Unnamed: 12'] == 'kbit/s':
        df_proposals.at[i,'Upload Data Rate'] *= 1e3
    if row['Units'] == 'GB':
        df_proposals.at[i,'Size'] *= 2**(30)
    elif row['Units'] == 'MB':
        df_proposals.at[i,'Size'] *= 2**(20)

#only keep certain columns for ease of use
df_proposals = df_proposals[value_cols]

#read requirements
df_requirements = pd.read_excel(proposals_filename, sheet_name = "Table 1 Design Proposals", header=14, nrows=7, usecols=["Property","Unit","Value","Limit"], index_col="Property")

#get names
df_admin = pd.read_excel(filepath, sheet_name='Submission', \
                    usecols=['Name','Section'], nrows=3)

#read table 2
df_table2 = pd.read_excel(filepath, sheet_name='Submission', \
                          usecols=['Design','(µV/level)', '(Hours)', '(minutes)', \
                                              'Total System Cost', 'Adjudication'], \
                          header=6, nrows=5)
#make table 2 column names readable
df_table2.rename(columns={'(µV/level)': 'ADC Resolution','(Hours)':'Storage Duration', \
                          '(minutes)':'Duration to Upload'}, inplace=True)
#get weights
df_weights = pd.read_excel(filepath, sheet_name='Submission', header=15, nrows=1, \
                           usecols=['Unnamed: 2', 'Unnamed: 5'])
#make weight columns readable
df_weights.rename(columns={'Unnamed: 2': 'Cost Weight', 'Unnamed: 5': 'MTBF Weight'}, inplace=True)
#read table 3
df_table3 = pd.read_excel(filepath, sheet_name='Submission', header=17, nrows=5, \
                          usecols=['Unnamed: 0','Raw','Norm','Weighted','Raw.1','Norm.1', \
                                   'Weighted.1','Total','Award Contract'])
#make design column readable
df_table3.rename(columns={'Unnamed: 0': 'Design'}, inplace=True)
#read table 4 (must read binary output as a string to prevent leading zeros from getting removed)
df_table4 = pd.read_excel(filepath, sheet_name='Submission', header=26, nrows=1, \
                          usecols="A:F", converters={'Binary Output': lambda x: str(x)})
df_table4.rename(columns={'Bias (mV)': 'Bias', 'fc/o (kHz)': 'fc/o', \
                          'Capacitance (nF)': 'Capacitance'}, inplace=True)

#run grader
grader = project2_grader(df_table2, df_weights, df_table3, df_table4, df_proposals, df_requirements)
eng_grade, dm_grade, grade_notes = grader.calc_grade(late)

output = {
    "output": '\n'.join(grade_notes),
    "visibility": "after_published", 
    "stdout_visibility": "hidden",
    "extra_data": {
        "Name": df_admin['Name'][0],
        "Section": df_admin['Section'][0],
        "Documentation": df_admin['Name'][2],
    },
    "tests":
    [{
        "score": eng_grade, 
        "max_score": 50, 
        "output": "Engineering Analysis",
        "tags": ["Engineering Analysis"], 
        "visibility": "after_published"
    },
    {
        "score": dm_grade, 
        "max_score": 25, 
        "output": "Decision Matrix",
        "tags": ["Decision Matrix"], 
        "visibility": "after_published"
    }]
}

with open(os.path.abspath('../results/results.json'), 'w') as f:
    json.dump(output, f)
