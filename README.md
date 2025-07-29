# Learning Behavior-Driven Knowledge Tracing Enhancement (LBE)

We propose LBE (Learning Behavior-Driven Knowledge Tracing Enhanced), a plug-and-play method for enhancing knowledge tracing through learning behavior analysis. This work focuses on characterizing and interpreting the relationship between learning behaviors and knowledge mastery

![FrameWork](./pic/framework.png)

The behavioral sequence incorporates three types of learning behaviors: answering time, number of attempts, and number of hints. Here, the *KT Model* in the Base Model refers to existing knowledge tracing models.



## Experiment Result
The overall performance comparison on three real-world datasets. Best results in bold. $^{\uparrow}$  indicates performance improved with LBE compared to the baseline. Baseline represents the base model $\mathcal{M}$ using LBE.

|      |    Datasets      | Assist09 |         |         | Assist12 |         |         |  Junyi  |         |         |
| :----------: | :------: | :------: | :-----: | :-----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: |
|    Models    |          |   AUC    |   ACC   |  RMSE   |   AUC    |   ACC   |  RMSE   |   AUC   |   ACC   |  RMSE   |
|  LBKT(2023)  |          | 0.77060  | 0.73782 | 0.42240 | 0.77519  | 0.75643 | 0.40859 | 0.79317 | 0.85309 | 0.33384 |
|  DKT (2015)  | baseline | 0.75454  | 0.73104 | 0.42586 | 0.73008  | 0.73505 | 0.42429 | 0.74878 | 0.84632 | 0.34421 |
|              |  LBE-I   | 0.76836<sup>↑</sup> | 0.73551<sup>↑</sup> | 0.42233<sup>↑</sup> | 0.76355<sup>↑</sup>  | 0.74850<sup>↑</sup> | 0.41379<sup>↑</sup> | 0.79947<sup>↑</sup> | 0.85659<sup>↑</sup> | 0.32979<sup>↑</sup> |
|              |  LBE-II  | 0.77741<sup>↑</sup> | 0.73877<sup>↑</sup> | 0.42024<sup>↑</sup> | 0.76551<sup>↑</sup>  | 0.74912<sup>↑</sup> | 0.41317<sup>↑</sup> | 0.79897<sup>↑</sup> | 0.85592<sup>↑</sup> | 0.32998<sup>↑</sup> |
| DKVMN (2017) | baseline | 0.75091  | 0.73009 | 0.42723 | 0.72851  | 0.73428 | 0.42477 | 0.74758 | 0.84693 | 0.34382 |
|              |  LBE-I   | 0.76978<sup>↑</sup>  | 0.73852<sup>↑</sup> | 0.42168<sup>↑</sup> | 0.76721<sup>↑</sup> | 0.75065<sup>↑</sup> | 0.41340<sup>↑</sup> | **0.80425**<sup>↑</sup> | **0.85728**<sup>↑</sup> | **0.32881**<sup>↑</sup> |
|              |  LBE-II  | 0.76779<sup>↑</sup>  | 0.73617<sup>↑</sup> | 0.42488<sup>↑</sup> | 0.76385<sup>↑</sup>  | 0.74912<sup>↑</sup> | 0.41398<sup>↑</sup> | 0.80326<sup>↑</sup> | 0.85708<sup>↑</sup> | 0.32907<sup>↑</sup> |
|  GKT (2019)  | baseline | 0.75601  | 0.73184 | 0.42585 | 0.73334  | 0.73470 | 0.42400 | 0.75167 | 0.84748 | 0.34293 |
|              |  LBE-I   | 0.76456<sup>↑</sup>  | 0.73256<sup>↑</sup> | 0.42333<sup>↑</sup> | 0.77225<sup>↑</sup>  | 0.75331<sup>↑</sup> | 0.41101<sup>↑</sup> | <u>0.80421</u><sup>↑</sup> | <u>0.85712</u><sup>↑</sup> | <u>0.32890</u><sup>↑</sup> |
|              |  LBE-II  | 0.76874<sup>↑</sup>  | 0.73582<sup>↑</sup> | 0.42432<sup>↑</sup> | 0.77139<sup>↑</sup>  | 0.75126<sup>↑</sup> | 0.41180<sup>↑</sup> | 0.80334<sup>↑</sup> | 0.85684<sup>↑</sup> | 0.32915<sup>↑</sup> |
| SAKT (2019)  | baseline | 0.73217  | 0.71823 | 0.43491 | 0.71384  | 0.72837 | 0.42984 | 0.74654 | 0.84573 | 0.34519 |
|              |  LBE-I   | 0.75341<sup>↑</sup>  | 0.72819<sup>↑</sup> | 0.43055<sup>↑</sup> | 0.75404<sup>↑</sup>  | 0.74403<sup>↑</sup> | 0.41777<sup>↑</sup> | 0.80058<sup>↑</sup> | 0.85586<sup>↑</sup> | 0.33045<sup>↑</sup> |
|              |  LBE-II  | 0.76524<sup>↑</sup>  | 0.73364<sup>↑</sup> | 0.42701<sup>↑</sup> | 0.75492<sup>↑</sup>  | 0.74469<sup>↑</sup> | 0.41824<sup>↑</sup> | 0.80049<sup>↑</sup> | 0.85565<sup>↑</sup> | 0.33095<sup>↑</sup> |
|  AKT (2020)  | baseline | 0.77939  | 0.74141 | 0.42249 | 0.77620  | 0.75568 | 0.40963 | 0.79103 | 0.85125 | 0.33383 |
|              |  LBE-I   | 0.79007<sup>↑</sup>  | **0.75168**<sup>↑</sup> | 0.41453<sup>↑</sup> | **0.78350**<sup>↑</sup> | **0.76113**<sup>↑</sup> | 0.40630<sup>↑</sup> | 0.80012<sup>↑</sup> | 0.85619<sup>↑</sup> | 0.33002<sup>↑</sup> |
|              |  LBE-II  | 0.79005<sup>↑</sup>  |    0.74956<sup>↑</sup>     | 0.41479<sup>↑</sup> | 0.78120<sup>↑</sup>  | 0.75949<sup>↑</sup> | 0.40719<sup>↑</sup> | 0.80029<sup>↑</sup> | 0.85599<sup>↑</sup> | 0.33018<sup>↑</sup> |
| MIKT (2024)  | baseline | 0.78670  | 0.74771 | 0.41437 | 0.77891  | 0.75855 | 0.40719 | 0.79582 | 0.85381 | 0.33239 |
|              |  LBE-I   | <u>0.79156</u><sup>↑</sup> | 0.75072<sup>↑</sup> | **0.41214**<sup>↑</sup> | <u>0.78314</u><sup>↑</sup> | <u>0.76048</u><sup>↑</sup> | **0.40583**<sup>↑</sup> | 0.79785<sup>↑</sup> | 0.85582<sup>↑</sup> | 0.33070<sup>↑</sup> |
|              |  LBE-II  | **0.79175**<sup>↑</sup> | <u>0.75079</u><sup>↑</sup> | <u>0.41265</u><sup>↑</sup> | 0.78157<sup>↑</sup>  | 0.75975<sup>↑</sup> | <u>0.40610</u><sup>↑</sup> | 0.80035<sup>↑</sup> | 0.85625<sup>↑</sup> | 0.32994<sup>↑</sup> |

where we denote the proposed LBE framework based on BQL-I as LBE-I and that based on BQL-II as LBE-II. 

