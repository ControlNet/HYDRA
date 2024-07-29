import json
import pandas as pd
import os
from torch.utils.data import Dataset
from rich.progress import track


class Refcoco(Dataset):

    def __init__(self, data_root: str) -> None:
        super().__init__()
        self.data_root = data_root
        # open test dataset
        with open(os.path.join(self.data_root, "testA.json")) as refcocotrain:
            ref_coco_train_data = json.load(refcocotrain)

        ref_df = pd.DataFrame(
            columns=['img_name', 'sent_id', 'sub_query', 'ground_true'])

        for img_set in track(ref_coco_train_data):

            # load each data
            img_name = img_set['img_name']

            ground_true = img_set['bbox']

            for sub in img_set['sentences']:

                # sub_query (content/question)
                sub_query = sub['sent']
                # query_id
                sent_id = sub['sent_id']

                new_row_ = pd.DataFrame.from_records(
                    [{'img_name': 'train2014/'+img_name, 'sent_id': sent_id, 'sub_query': sub_query, 'ground_true': ground_true}])
                ref_df = pd.concat([ref_df, new_row_], ignore_index=True)
        self.metadata = ref_df
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        return os.path.join(self.data_root, row.img_name), row.sent_id, row.sub_query, row.ground_true
    
    def __len__(self):
        return len(self.metadata)
    

class OKVQA(Dataset):
    def __init__(self, data_root) -> None:
        super().__init__()
        self.data_root = data_root
        question_path = os.path.join(data_root, 'OpenEnded_mscoco_val2014_questions.json')
        annotation_path = os.path.join(data_root, 'mscoco_val2014_annotations.json')

        # question_path = '../refer_cocodataset/okvqa_dataset/OpenEnded_mscoco_val2014_questions.json'
        with open(question_path) as okvqa_dataset:
            okvqa_question_data = json.load(okvqa_dataset)

        # annotation_path = '../refer_cocodataset/okvqa_dataset/mscoco_val2014_annotationsv11.json'
        with open(annotation_path) as okvqa_annotation:
            okvqa_annotation_data = json.load(okvqa_annotation)

        okvqa_df = pd.DataFrame(
            columns=['img_name', 'sent_id', 'sub_query', 'ground_true'])

        for each_question in okvqa_question_data["questions"]:
            img_name = 'val2014/COCO_val2014_{:0>12}'.format(
                str(each_question['image_id']))+'.jpg'
            sent_id = each_question['question_id']
            sub_query = each_question['question']
            ground_true_answer_list = [anno_[
                "answers"] for anno_ in okvqa_annotation_data["annotations"] if anno_["question_id"] == sent_id][0]
            ground_true_list = set([anno_["answer"]
                                for anno_ in ground_true_answer_list])

            new_row_ = pd.DataFrame.from_records(
                [{'img_name': img_name, 'sent_id': sent_id, 'sub_query': sub_query, 'ground_true': ground_true_list}])
            okvqa_df = pd.concat([okvqa_df, new_row_], ignore_index=True)
        self.metadata = okvqa_df

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        return os.path.join(self.data_root, row.img_name), row.sent_id, row.sub_query, list(row.ground_true)
    
    def __len__(self):
        return len(self.metadata)
    

class AOKVQA(Dataset):

    def __init__(self, aokvqa_dir, split, coco_dir, version='v1p0') -> None:
        super().__init__()
        assert split in ['train', 'val', 'test', 'test_w_ans']
        dataset = json.load(open(
            os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
        ))
        dataset = pd.DataFrame(dataset)
        dataset['img_name'] = dataset.apply(
            lambda row: self._get_coco_path(split, row['image_id'], coco_dir), axis=1)
        dataset.rename(columns={'question_id': 'sent_id'}, inplace=True)
        dataset.rename(columns={'question': 'sub_query'}, inplace=True)
        dataset.rename(columns={'direct_answers': 'ground_true'}, inplace=True)
        self.metadata = dataset

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        return row.img_name, row.sent_id, row.sub_query, row.ground_true
    
    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def _get_coco_path(split, image_id, coco_dir):
        return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")
