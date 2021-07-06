import os
import re
import pdfplumber
import sys
import pdfplumber
import os
from datetime import datetime
import pandas as pd
from os.path import dirname as up
# one_up = up(__file__)
# sys.path.insert(0, one_up)
#
# sys.path.insert(1, 'E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core')
# # sys.path.insert(1, 'E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core\\src\\')
#
# import pickle
# import src.core.emotions.emotion_extractor as emotion_extractor
# import src.utils.text_processor as text_utils
# import src.core.summary.keyphrase_extractor as keyphrase_extractor
# import src.core.clinical_info.clinical_info_extractor as clinical_info_extractor
from transformers import AutoTokenizer, AutoModel

from emotion_candidate_recognition import sum_up_dicts
from predict_for_documents import emo_detect_document_lnm


def process_pdf(f_path):
    tokenizer = AutoTokenizer.from_pretrained("E:\Projects\emo_detector_new\emo_bert_model")
    model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\emo_bert_model")
    presentation_body = ''
    qa_body = ''
    pres_segments = []
    qa_segments = []
    all_participants = []
    keyphrases_pb = []
    keyphrases_qab = []
    emotion_profiles_for_presentation_segments = []
    emotion_profiles_for_qa_segments = []
    e_p_presentation_body = []
    e_p_qa = []
    processed_data = {}
    pdf = pdfplumber.open(f_path)
    full_text = ''
    total_pages = len(pdf.pages)

    cover_page = pdf.pages[0].extract_text()
    cv_text_a = cover_page.split('PLACEHOLDER TRANSCRIPT')[1]
    cv_text_b = cv_text_a.split('Earnings')[0]
    cv_text_b_splits = cv_text_b.split(' ')

    year_cv = cv_text_b_splits[1].strip()
    qtr_cv = cv_text_b_splits[0].strip()
    # print(year_cv,qtr_cv)
    comp_name = (" ").join(cv_text_b_splits[2:])
    print("year",year_cv)
    print("quater",qtr_cv)
    print("comp_name",comp_name)
    print('********************')

    for each_page in range(1,total_pages):
        page = pdf.pages[each_page]
        text = page.extract_text()
        text+='\n\n\n'
        full_text+=text

    # print(full_text)

    participants = []
    second_page = pdf.pages[1]
    participant_text = second_page.extract_text()
    all_participants = ['Operator']
    print('processing participants')
    if('CORPORATE PARTICIPANTS' in participant_text and 'PRESENTATION' in participant_text):
        part_a = participant_text.split('CORPORATE PARTICIPANTS')[1]
        part_b = part_a.split('CONFERENCE CALL PARTICIPANTS')
        cop_parti = part_b[0].strip()
        conf_parti = part_b[1].split('PRESENTATION')[0].strip()
        # print('CORPORATE PARTICIPANTS')
        # print(cop_parti)
        # print('CONFERENCE CALL PARTICIPANTS')
        # print(conf_parti)
        cop_parti_list = cop_parti.split('\n')
        conf_parti_list = conf_parti.split('\n')
        # print(cop_parti_list)
        # print(conf_parti_list)
        all_participants.extend(cop_parti_list)
        all_participants.extend(conf_parti_list)


    if('PRESENTATION' in full_text and 'QUESTIONS AND ANSWERS' in full_text):
        presentation_first = full_text.split('PRESENTATION')[1]
        body = presentation_first.split('QUESTIONS AND ANSWERS')
        presentation_body= body[0]
        pb = presentation_body

        # emotion_profile_pb = emo_detecct_document(pb)

        # e_p_presentation_body = emotion_profile_pb
        qa_body = body[1].split('DISCLAIMER')[0]
        qab = qa_body

        # emotion_profile_qab = emo_detecct_document(qab)

        # e_p_qa = emotion_profile_qab
        print('presentation body>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print(presentation_body[:20])
        print('qa body>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print(qa_body[:20])

        print('locating people in the presentation body')
        occurances = []
        for p in all_participants:
            # print(p)
            if p in presentation_body:
                # print(p)
                occ_list = [m.start() for m in re.finditer(p, presentation_body)]
                # print(presentation_body.find(p))
                if (p == 'Operator'):
                    occurances.append([occ_list[0],p])
                else:
                    for each_o in occ_list:
                        occurances.append([each_o,p])
        occurances = sorted(occurances, key=lambda x: x[0])
        occurances.append([len(presentation_body),'End'])


        for i in range(len(occurances)-1):
            # print('>>>>>>>>>>>>>>>>>')
            # print(occurances[i],occurances[i+1])
            body_part = presentation_body[occurances[i][0]:occurances[i+1][0]]
            # print(body_part)
            text_1 = body_part

            emotion_profile= emo_detect_document_lnm(text_1,tokenizer,model,10)
            pres_segments.append([occurances[i][1]]+[[emotion_profile]])
            emotion_profiles_for_presentation_segments.append(emotion_profile)



        # print(occurances)

        print('locating people in the qa body')
        qa_occurances = []
        for p in all_participants:
            # print(p)
            if p in qa_body:
                # print(p)
                qa_occ_list = [m.start() for m in re.finditer(p, qa_body)]
                # print(presentation_body.find(p))
                if (p == 'Operator'):
                    continue
                else:
                    for each_o in qa_occ_list:
                        qa_occurances.append([each_o, p])
        qa_occurances = sorted(qa_occurances, key=lambda x: x[0])
        qa_occurances.append([len(qa_body), 'End'])


        for i in range(len(qa_occurances) - 1):
            # print('>>>>>>>>>>>>>>>>>')
            # print(qa_occurances[i], qa_occurances[i + 1])
            qa_body_part = qa_body[qa_occurances[i][0]:qa_occurances[i + 1][0]]
            # print(qa_body_part)
            text_1 = qa_body_part

            emotion_profile = emo_detect_document_lnm(text_1,tokenizer,model,3)
            qa_segments.append(qa_occurances[i] + [emotion_profile])
            emotion_profiles_for_qa_segments.append(emotion_profile)

    pdf.close()

    processed_data['pdf'] = f_path.split('\\')[-1]
    processed_data['full_text'] = full_text
    processed_data['participants'] = all_participants
    processed_data['presentation_body'] = presentation_body
    processed_data['qa_body'] = qa_body
    processed_data['presentation_segments'] = pres_segments
    processed_data['e_p_pb'] = sum_up_dicts(emotion_profiles_for_presentation_segments)
    processed_data['e_p_qa'] = sum_up_dicts(emotion_profiles_for_qa_segments)
    processed_data['qa_segments'] = qa_segments
    processed_data['emotion_profiles_pres'] = emotion_profiles_for_presentation_segments
    processed_data['emotion_profiles_qa'] = emotion_profiles_for_qa_segments
    processed_data['comp_name']= comp_name
    processed_data['year'] = year_cv
    processed_data['quarter'] = qtr_cv
    print(processed_data)
    return processed_data


import csv
if __name__ == '__main__':
    f = open('errors.txt','w')

    with open(r'E:\Projects\emo_detector_new\transcript_results\pros_all_meta_reports_lnm.csv', mode='w',newline='',encoding='utf-8') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        employee_writer.writerow(['pdf','participants', '#presentation_body', '#qa_body','presentation_all_e_p','qa_all_e_p', '#presentation_segments', '#qa_segments','presentation_segments', 'qa_segments','e_p_pres','e_p_qa',"year","quarter","comp_name"])

        pdf_list = os.listdir("E:\Data\Emotion_detection_gihan\earningCalls-2010-2012\\2011 Jul to Dec\\")
        i=0
        for each_file in pdf_list:
            tt1 = datetime.now()
            i=i+1
            if(each_file in ['2011-Aug-03-BEN.N-140511442202-transcript.pdf','2011-Aug-05-ATO.N-137689101588-transcript.pdf','2011-Aug-16-WMT.N-139819905465-transcript.pdf']):continue
            path_pdf = os.path.join("E:\Data\Emotion_detection_gihan\earningCalls-2010-2012\\2011 Jul to Dec",each_file)
            print(path_pdf)
            try:
                output = process_pdf(path_pdf)
                # employee_writer.writerow([str(output['pdf']),str(output['participants']),str(output['presentation_body']),str(output['qa_body']),str(output['presentation_segments']),str(output['qa_segments'])])

                employee_writer.writerow([str(output['pdf']), str(output['participants']),[len(output['presentation_body'])],[len(output['qa_body'])],output['e_p_pb'],output['e_p_qa'],[len(output['presentation_segments'])],[len(output['qa_segments'])],output['presentation_segments'],output['qa_segments'],output['emotion_profiles_pres'],output['emotion_profiles_qa'],output["year"],output["quarter"],output["comp_name"]])
            except SyntaxError:
                f.write(str(path_pdf)+'\n')
            tt2 = datetime.now()
            diff = tt2 - tt1
            print('time', diff)

            if(i==1):break


    employee_file.close()
    f.close()
    #
    # for k in output:
    #     print('**********')
    #     print(k)
    #     print(output[k])







