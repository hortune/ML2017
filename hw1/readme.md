# Best grade
## Result 
| learning rate | type | iter | loss | final|sample|
|:------:|:------:|:------:|:------:|:------:|:------:|
|1e -11| constant | 20000 | NaN | 13% |172|
|1e -8 | constat|20000|4000|7.5%|172|
|2.00506e-9|constant|20000|3888|8.168016%|172|
|1e-6/172|adagrad|20000|NaN|7.8%|172|
|500|adagrad|200000|2900|TODO|172|
|500|adagrad|200000|5811|TODO|mode 1|
|1e-8|fit|200000|6140|TODO|mode 2|
|500|adagrad|200000|6140|TODO|mode 1|
|500|adagrad|1000000|2391|TODO|172|
|500|adagrad|5000000|1997|TODO|172|
## mode
| name | PM2.5 | PM10 | quad PM2.5 | quad PM10 | O3 | RAINFALL|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|mode 1|9|5|4|2|3|2|
|mode 2|9|9|0|0|0|0|


## Result with increase_data and validation
|learning rate|type|iter|rmse|final|sample|
|:------:|:------:|:------:|:------:|:------:|:------:|
|0.030517578125|adagrad|10000|7.5x|TODO|172|
|0.06103515625|adagrad|10000|7.67613663552|TODO|4*18|
|0.003814697|adagrad|50000|5.924945855|5.97557|mode 1|

