# Data explanation

This file explains the data which are provided to the challenge *Day-ahead Active Losses Forecasting*. The raw data will be provided as time series data between 2019 and 2021.

As the active losses data in 2022 is not published till now. Please use the data in 2019-2021 to train, validate, and test your model. When more data is available in the future, we can update it here.

The target variable is "Active-Losses". The goal is to forecast the active losses for the next day in hourly resolution, meaning the output should be 24 numbers (multi-step forecasting output). You have the freedom to choose any data as feature(s) for your model. For example, it is recommended to use the active losses in the past 7 days as one of the features, because of the autocorrelation and the weekly seasonality in the active losses data. However, please choose the feature(s) and the length of the inputs as you wish. 

Below are the details of each dataset:

* Active-losses: historical active losses data, unit: kWh per 15 minutes

    * Convert to **beginning** of the timestamp 
    
    The time is given is the **end** of each timestamp, meaning the timestamp "2019-01-01 00:15:00" records the data for the time slot between *2019-01-01 00:00:00* and *2019-01-01 00:15:00*. You need to shift every timestamp -15 minutes to be in line with the common practice that the timestamp corresponds to the **beginning** of the timestamp. 

    * 15-min resolution

    This dataset is in 15-min resolution, you need t convert them to hourly resolution by aggregating four 15-min data within each hour. 

    ** Convert to MWh

    You need to convert **kWh** to **MWh** by dividing it by 1000. As all other variables use MW. 

* Forecast-renewable-generation (unit: MW)

    * solar_fore_de [MW]: forecast of solar generation in Germany
    * solar_fore_it [MW]: forecast of solar generation in Italy
    * wind_fore_de [MW]: forecast of wind generation in Germany
    * wind_fore_de [MW]: forecast of wind generation in Germany

* Forecast-temperature (unit: °C)

    This dataset is in 6-hour resolution, you need to preprocess it and convert it to hourly resolution if you decide to use them.

    * temperature_fore_ch: temperature forecast of Switzerland
    * temperature_fore_fr: temperature forecast of France
    * temperature_fore_de: temperature forecast of Germany
    * temperature_fore_it: temperature forecast of Italy

* NTC (unit: MW)

    NTC (Net Transfer Capacity) is the maximum exchange programme between two areas which is consistent with the security standards of both areas (*simply put, these are the planned maximun cross-border capacity that electricity can flow from one country to another*). Below are the NTC between Swissgrid and its neighboring TSOs (Transmission System Operators). 

    * CH_AT: the maximum exchange programme from CH to AT
    * CH_DE: the maximum exchange programme from CH to DE
    * CH_FR: the maximum exchange programme from CH to FR
    * CH_IT: the maximum exchange programme from CH to IT
    * AT_CH: the maximum exchange programme from AT to CH
    * DE_CH: the maximum exchange programme from DE to CH
    * FR_CH: the maximum exchange programme from FR to CH
    * IT_CH: the maximum exchange programme from IT to CH

The reason these datasets are chosen for this challenge are:

* Active losses is a timeseries data with the characteristics of autocorrelation and seasonality
* High renewable generation in neighboring countries tend to decrease the local electricity price and increase the export to its neighboring countries. Located next to Germany and Italy, Switzerland can have higher import or bypassing flows when Germany and/or Italy have high renewable energy generation.
* Temperature can influence the electricity demand, and thus the losses happening on the grid.
* NTC limits how much cross-board flows Switzerland can have with its neighboring countries, the lower the limit, the lower the flow, and thus lower losses. 

In summary, all these factors can be relevant for forecasting the active losses, please find out which can create the best-performed model. 

A small note: in case of any data missing for some timestamps, which is common when working with the real world data - please clean it as needed.

Good luck and have fun!
