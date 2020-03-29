#!/usr/bin/env bash

mysqldump -u root -h at-database.ad.bcm.edu -p --databases  nips2018_analysis_performance nips2018_analysis_tuning nips2018_data nips2018_models nips2018_oracle nips2018_parameters > nips2018.sql