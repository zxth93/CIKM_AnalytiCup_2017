# CIKM_AnalytiCup_2017
This repo discribes the solution of Team 怀北村明远湖. [CIKM AnalytiCup 2017](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.773ef42fBB8Kok&raceId=231596) is an open competition that is sponsored by Shenzhen Meteorological Bureau, Alibaba Group and CIKM2017. Our team got the third place in the first phrase. And in the second phrase we got the fourth place.

# Introduction
Short-term precipitation forecasting such as rainfall prediction is a task to predict a short-term rainfall amount based on current observations. In this challenge, sponsors provide a set of radar maps at different time spans where each radar map covers radar reflectivity of a target site and its surrounding areas. Radar maps are measured at different time spans, i.e., 15 time spans with an interval of 6 minutes, and different heights, i.e., 4 heights, from 0.5km to 3.5km with an interval of 1km; Each radar map covers an area of 101km\*101km around the site. The area is marked as 101\*101 grids, and the target site is located at the centre, i.e. (50, 50).

Our task here is to predict the total rainfall amount on the ground between future 1-hour and 2-hour for each target site.In this challenge, we combine Random Forestry, XGBoost and Bidirectional Gated Recurrent Units (GRUs) into an ensemble model to tackle this problem and achieve satisfying result.

![](https://img.alicdn.com/tps/TB1mmZRPFXXXXaPaXXXXXXXXXXX-865-302.png)

# Data Process



