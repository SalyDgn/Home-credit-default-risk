"""Project parameters."""

PARAMS = {
    'file_directory': "data/home-credit-default-risk/",
    'cleaned_data_directory': "data/home-credit-default-risk/cleaned_data/",
}

MODEL_PARAMS = {
    "TARGET_NAME": "TARGET",
    "features_selected": [
        "CNT_CHILDREN",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "REGION_POPULATION_RELATIVE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_PHONE",
        "CNT_FAM_MEMBERS",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "HOUR_APPR_PROCESS_START",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "APARTMENTS_AVG",
        "BASEMENTAREA_AVG",
        "YEARS_BEGINEXPLUATATION_AVG",
        "YEARS_BUILD_AVG",
        "COMMONAREA_AVG",
        "ELEVATORS_AVG",
        "ENTRANCES_AVG",
        "FLOORSMAX_AVG",
        "FLOORSMIN_AVG",
        "LANDAREA_AVG",
        "LIVINGAPARTMENTS_AVG",
        "LIVINGAREA_AVG",
        "NONLIVINGAREA_AVG",
        "APARTMENTS_MODE",
        "BASEMENTAREA_MODE",
        "YEARS_BEGINEXPLUATATION_MODE",
        "YEARS_BUILD_MODE",
        "COMMONAREA_MODE",
        "ELEVATORS_MODE",
        "ENTRANCES_MODE",
        "FLOORSMAX_MODE",
        "FLOORSMIN_MODE",
        "LANDAREA_MODE",
        "LIVINGAPARTMENTS_MODE",
        "LIVINGAREA_MODE",
        "NONLIVINGAREA_MODE",
        "APARTMENTS_MEDI",
        "BASEMENTAREA_MEDI",
        "YEARS_BEGINEXPLUATATION_MEDI",
        "YEARS_BUILD_MEDI",
        "COMMONAREA_MEDI",
        "ELEVATORS_MEDI",
        "ENTRANCES_MEDI",
        "FLOORSMAX_MEDI",
        "FLOORSMIN_MEDI",
        "LANDAREA_MEDI",
        "LIVINGAPARTMENTS_MEDI",
        "LIVINGAREA_MEDI",
        "NONLIVINGAREA_MEDI",
        "TOTALAREA_MODE",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "DAYS_LAST_PHONE_CHANGE",
        "FLAG_DOCUMENT_3",
        "FLAG_DOCUMENT_6",
        "FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14",
        "FLAG_DOCUMENT_16",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "NAME_CONTRACT_TYPE_Cashloans",
        "NAME_CONTRACT_TYPE_Revolvingloans",
        "CODE_GENDER_F",
        "CODE_GENDER_M",
        "FLAG_OWN_CAR_N",
        "FLAG_OWN_CAR_Y",
        "NAME_TYPE_SUITE_Family",
        "NAME_INCOME_TYPE_Commercialassociate",
        "NAME_INCOME_TYPE_Pensioner",
        "NAME_INCOME_TYPE_Stateservant",
        "NAME_INCOME_TYPE_Working",
        "NAME_EDUCATION_TYPE_Highereducation",
        "NAME_EDUCATION_TYPE_Lowersecondary",
        "NAME_EDUCATION_TYPE_Secondarysecondaryspecial",
        "NAME_FAMILY_STATUS_Civilmarriage",
        "NAME_FAMILY_STATUS_Married",
        "NAME_FAMILY_STATUS_Singlenotmarried",
        "NAME_FAMILY_STATUS_Widow",
        "NAME_HOUSING_TYPE_Houseapartment",
        "NAME_HOUSING_TYPE_Rentedapartment",
        "NAME_HOUSING_TYPE_Withparents",
        "OCCUPATION_TYPE_Accountants",
        "OCCUPATION_TYPE_Cookingstaff",
        "OCCUPATION_TYPE_Corestaff",
        "OCCUPATION_TYPE_Drivers",
        "OCCUPATION_TYPE_Highskilltechstaff",
        "OCCUPATION_TYPE_Laborers",
        "OCCUPATION_TYPE_LowskillLaborers",
        "OCCUPATION_TYPE_Managers",
        "OCCUPATION_TYPE_Salesstaff",
        "OCCUPATION_TYPE_Securitystaff",
        "OCCUPATION_TYPE_Waitersbarmenstaff",
        "ORGANIZATION_TYPE_Agriculture",
        "ORGANIZATION_TYPE_Bank",
        "ORGANIZATION_TYPE_BusinessEntityType3",
        "ORGANIZATION_TYPE_Construction",
        "ORGANIZATION_TYPE_Government",
        "ORGANIZATION_TYPE_Industrytype3",
        "ORGANIZATION_TYPE_Medicine",
        "ORGANIZATION_TYPE_Military",
        "ORGANIZATION_TYPE_Police",
        "ORGANIZATION_TYPE_Restaurant",
        "ORGANIZATION_TYPE_School",
        "ORGANIZATION_TYPE_Security",
        "ORGANIZATION_TYPE_SecurityMinistries",
        "ORGANIZATION_TYPE_Selfemployed",
        "ORGANIZATION_TYPE_Tradetype3",
        "ORGANIZATION_TYPE_Transporttype3",
        "ORGANIZATION_TYPE_XNA",
        "FONDKAPREMONT_MODE_orgspecaccount",
        "FONDKAPREMONT_MODE_regoperaccount",
        "FONDKAPREMONT_MODE_regoperspecaccount",
        "HOUSETYPE_MODE_blockofflats",
        "WALLSMATERIAL_MODE_Monolithic",
        "WALLSMATERIAL_MODE_Panel",
        "WALLSMATERIAL_MODE_Stonebrick",
        "WALLSMATERIAL_MODE_Wooden",
        "EMERGENCYSTATE_MODE_No",
        "BUREAU_BUREAU_DAYS_CREDIT",
        "BUREAU_BUREAU_DAYS_CREDIT_ENDDATE",
        "BUREAU_BUREAU_DAYS_ENDDATE_FACT",
        "BUREAU_BUREAU_AMT_CREDIT_SUM",
        "BUREAU_BUREAU_AMT_CREDIT_SUM_OVERDUE",
        "BUREAU_BUREAU_DAYS_CREDIT_UPDATE",
        "BUREAU_BUREAU_MONTHS_BALANCE",
        "PREV_AMT_ANNUITY",
        "PREV_AMT_APPLICATION",
        "PREV_AMT_CREDIT",
        "PREV_AMT_DOWN_PAYMENT",
        "PREV_AMT_GOODS_PRICE",
        "PREV_HOUR_APPR_PROCESS_START",
        "PREV_NFLAG_LAST_APPL_IN_DAY",
        "PREV_RATE_DOWN_PAYMENT",
        "PREV_DAYS_DECISION",
        "PREV_CNT_PAYMENT",
        "PREV_DAYS_FIRST_DUE",
        "PREV_DAYS_LAST_DUE_1ST_VERSION",
        "PREV_DAYS_LAST_DUE",
        "PREV_DAYS_TERMINATION",
        "PREV_PREV_APP_COUNT",
        "PREV_NAME_CONTRACT_TYPE_Cashloans",
        "PREV_NAME_CONTRACT_TYPE_Consumerloans",
        "PREV_NAME_CONTRACT_TYPE_Revolvingloans",
        "PREV_WEEKDAY_APPR_PROCESS_START_MONDAY",
        "PREV_FLAG_LAST_APPL_PER_CONTRACT_Y",
        "PREV_NAME_CASH_LOAN_PURPOSE_Buildingahouseoranannex",
        "PREV_NAME_CASH_LOAN_PURPOSE_Carrepairs",
        "PREV_NAME_CASH_LOAN_PURPOSE_Medicine",
        "PREV_NAME_CASH_LOAN_PURPOSE_Other",
        "PREV_NAME_CASH_LOAN_PURPOSE_Paymentsonotherloans",
        "PREV_NAME_CASH_LOAN_PURPOSE_Repairs",
        "PREV_NAME_CASH_LOAN_PURPOSE_Urgentneeds",
        "PREV_NAME_CONTRACT_STATUS_Approved",
        "PREV_NAME_CONTRACT_STATUS_Canceled",
        "PREV_NAME_CONTRACT_STATUS_Refused",
        "PREV_NAME_PAYMENT_TYPE_Cashthroughthebank",
        "PREV_NAME_PAYMENT_TYPE_XNA",
        "PREV_CODE_REJECT_REASON_HC",
        "PREV_CODE_REJECT_REASON_LIMIT",
        "PREV_CODE_REJECT_REASON_SCO",
        "PREV_CODE_REJECT_REASON_SCOFR",
        "PREV_CODE_REJECT_REASON_XAP",
        "PREV_NAME_TYPE_SUITE_Children",
        "PREV_NAME_TYPE_SUITE_Family",
        "PREV_NAME_CLIENT_TYPE_New",
        "PREV_NAME_CLIENT_TYPE_Refreshed",
        "PREV_NAME_GOODS_CATEGORY_ClothingandAccessories",
        "PREV_NAME_GOODS_CATEGORY_ConstructionMaterials",
        "PREV_NAME_GOODS_CATEGORY_ConsumerElectronics",
        "PREV_NAME_GOODS_CATEGORY_Furniture",
        "PREV_NAME_GOODS_CATEGORY_MedicalSupplies",
        "PREV_NAME_GOODS_CATEGORY_Mobile",
        "PREV_NAME_GOODS_CATEGORY_XNA",
        "PREV_NAME_PORTFOLIO_Cards",
        "PREV_NAME_PORTFOLIO_Cash",
        "PREV_NAME_PORTFOLIO_POS",
        "PREV_NAME_PORTFOLIO_XNA",
        "PREV_NAME_PRODUCT_TYPE_XNA",
        "PREV_NAME_PRODUCT_TYPE_walkin",
        "PREV_NAME_PRODUCT_TYPE_xsell",
        "PREV_CHANNEL_TYPE_APCashloan",
        "PREV_CHANNEL_TYPE_Channelofcorporatesales",
        "PREV_CHANNEL_TYPE_Contactcenter",
        "PREV_CHANNEL_TYPE_Creditandcashoffices",
        "PREV_CHANNEL_TYPE_Stone",
        "PREV_NAME_SELLER_INDUSTRY_Clothing",
        "PREV_NAME_SELLER_INDUSTRY_Connectivity",
        "PREV_NAME_SELLER_INDUSTRY_Construction",
        "PREV_NAME_SELLER_INDUSTRY_Consumerelectronics",
        "PREV_NAME_SELLER_INDUSTRY_Furniture",
        "PREV_NAME_SELLER_INDUSTRY_XNA",
        "PREV_NAME_YIELD_GROUP_XNA",
        "PREV_NAME_YIELD_GROUP_high",
        "PREV_NAME_YIELD_GROUP_low_action",
        "PREV_NAME_YIELD_GROUP_low_normal",
        "PREV_NAME_YIELD_GROUP_middle",
        "PREV_PRODUCT_COMBINATION_CardStreet",
        "PREV_PRODUCT_COMBINATION_CardXSell",
        "PREV_PRODUCT_COMBINATION_Cash",
        "PREV_PRODUCT_COMBINATION_CashStreethigh",
        "PREV_PRODUCT_COMBINATION_CashStreetmiddle",
        "PREV_PRODUCT_COMBINATION_CashXSellhigh",
        "PREV_PRODUCT_COMBINATION_CashXSelllow",
        "PREV_PRODUCT_COMBINATION_CashXSellmiddle",
        "PREV_PRODUCT_COMBINATION_POShouseholdwithoutinterest",
        "PREV_PRODUCT_COMBINATION_POSindustrywithinterest",
        "PREV_PRODUCT_COMBINATION_POSindustrywithoutinterest",
        "PREV_PRODUCT_COMBINATION_POSmobilewithinterest",
        "INSTA_NUM_INSTALMENT_VERSION",
        "INSTA_DAYS_INSTALMENT",
        "INSTA_DAYS_ENTRY_PAYMENT",
        "INSTA_AMT_INSTALMENT",
        "INSTA_AMT_PAYMENT",
        "POS_MONTHS_BALANCE",
        "POS_CNT_INSTALMENT",
        "POS_CNT_INSTALMENT_FUTURE",
        "POS_NAME_CONTRACT_STATUS_Active",
        "POS_NAME_CONTRACT_STATUS_Returnedtothestore",
        "POS_NAME_CONTRACT_STATUS_Signed",
        "CC_MONTHS_BALANCE",
        "CC_AMT_BALANCE",
        "CC_AMT_DRAWINGS_ATM_CURRENT",
        "CC_AMT_DRAWINGS_CURRENT",
        "CC_AMT_DRAWINGS_POS_CURRENT",
        "CC_AMT_INST_MIN_REGULARITY",
        "CC_AMT_PAYMENT_CURRENT",
        "CC_AMT_PAYMENT_TOTAL_CURRENT",
        "CC_AMT_RECEIVABLE_PRINCIPAL",
        "CC_AMT_RECIVABLE",
        "CC_AMT_TOTAL_RECEIVABLE",
        "CC_CNT_DRAWINGS_ATM_CURRENT",
        "CC_CNT_DRAWINGS_CURRENT",
        "CC_CNT_DRAWINGS_OTHER_CURRENT",
        "CC_CNT_DRAWINGS_POS_CURRENT",
        "CC_NAME_CONTRACT_STATUS_Active",
    ],
}

SEED = 42
