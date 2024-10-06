You can download the refined test dataset from google drive and refinement code is in /refinement/refine.py:

https://drive.google.com/file/d/1OuYnLXYoRp6i6jdID3ntdNg9A0rgFcoc/view?usp=drive_link

Then extract the zip, all the content should be directly in this data folder.

Please ensure that your data has the following format:

```
├──data
│   ├── HMDB51
│   ├── OCL
│   │   ├── COCO
│   │   ├── from_web
│   │   └── ImageNet
│   ├── Pangea
│   ├── B123_test_KIN-FULL_with_node.pkl
│   ├── HMDB.pkl
│   ├── OCL_annot_test.pkl
│   ├── OCL_selected_test_affordance_refined_new_1.pkl
│   ├── OCL_selected_test_attribute_refined_new_1.pkl
│   ├── pangea_test_refined_new.pkl
```