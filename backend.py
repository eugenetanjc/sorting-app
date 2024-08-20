# Importing required libraries
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import psycopg2
import warnings
warnings.filterwarnings("ignore")

def sorting(s_country, year, s_week, s_ctype, params_dict):

    # Processing user input
    country = "".join([str(current_country) for current_country in s_country])
    week = int("".join([str(current_integer) for current_integer in s_week]))
    countrytype = "".join([str(current_ctype) for current_ctype in s_ctype])
    year = int(year)

    for var in [country, year, week, countrytype]:    
        print(var, type(var))

    # Get date related information
    current_date = datetime.now()
    current_week_number = current_date.isocalendar()[1]
    current_season_number = current_week_number // 13

    current_year = str(current_date.year)
    season_start_month = f"{(current_season_number * 3 + 1):02d}"

    start_of_week = (current_date - timedelta(days=current_date.weekday())).strftime("%Y-%m-%d")

    # Reading key size check parameters from Parameters worksheet
    param_df = params_dict['Parameters'].iloc[:, 0:2]
    param_dict = param_df.set_index("Parameter")["Value"].to_dict()
    Min_SOH_per_Article_Col = param_dict['Ideal OH Per Article-Color']
    Min_Num_KeySize_per_Article_Col = param_dict['Minimum Number of Key Sizes Per Article-Color']
    Min_SOH_Per_KeySize_per_Article_Col = param_dict['Minimum OH Per Key Sizes Per Article-Color']
    Min_Num_Size_with_min_OH_per_Article_Col = param_dict['Ideal Number of Size with Minimum OH Per Article-Color']
    Min_Num_Colors_per_Article = param_dict['Minimum Number of Colors to Pass Criteria Per Article']

    # Selecting the parameters for key size info, and reading calendar info from Seasonal Calendar worksheet
    if countrytype == 'Hot':
        key_size_cols = slice(4,6)
        calendar_df = params_dict["Seasonal Calendar"].iloc[3:55, 0:9]
        calendar_df.columns = ["Month","week","Open Toe","Platform Open Toe","Covered Open-Back",
                            "Platform Covered Open-Back","Covered","Platform Covered","Boots"]
        
    elif countrytype == 'Cold':
        key_size_cols = slice(7, 9)  
        calendar_df = params_dict["Seasonal Calendar"].iloc[3:55, 10:19]
        calendar_df.columns = ["Month","week","Boots","Covered","Platform Covered","Covered Open-Back", 
                               "Platform Covered Open-Back","Open Toe","Platform Open Toe"]
    else:
        key_size_cols = slice(10, 12)  
        calendar_df = params_dict["Seasonal Calendar"].iloc[3:55, 24:33]
        calendar_df.columns = ["Month","week","Boots","Covered","Platform Covered","Covered Open-Back",
                                "Platform Covered Open-Back","Open Toe","Platform Open Toe"]

    key_sizes_df = params_dict["Parameters"].iloc[1:8, key_size_cols]
    key_sizes_df.columns = ['Size', 'Key Size']

    # Reading SOH info from SOH worksheet
    soh_df = params_dict["SOH"].iloc[0:, 0:4]
    soh_df.columns = ["Article","SOH","Colour","Size"]
    
    # Using key size info and SOH info to perform KeySizeCheck Filtering
    KeySizeCheck_df = soh_df.copy()
    KeySizeCheck_df['Is Key Size'] = KeySizeCheck_df['Size'].map(key_sizes_df.dropna().set_index('Size')['Key Size']).fillna(False).astype(bool)

    # Aggregate SOH by Article-Color, and count number of key sizes
    KeySizeCheck_agg = KeySizeCheck_df.groupby(['Article','Colour'],as_index=False).agg({'SOH':'sum','Is Key Size':'sum'})
    KeySizeCheck_agg = KeySizeCheck_agg.rename(columns={'SOH': 'Article Color SOH',
                                                        'Is Key Size': 'Article Color Num Key Sizes'})
    
    # KeySizeCheck Step 1: Check if total SOH for each Article-Color is above the minimum
    KeySizeCheck_df = pd.merge(KeySizeCheck_df, KeySizeCheck_agg[['Article', 'Colour', 'Article Color SOH', 'Article Color Num Key Sizes']], on=['Article','Colour'], how='left')
    KeySizeCheck_df["Key Size Check 1"] = (KeySizeCheck_df["Article Color SOH"] >= Min_SOH_per_Article_Col).astype(int)

    # KeySizeCheck Step 2: Check if total SOH for each Article-Color-KeySize is above the minimum
    KeySizeCheck_keysize_agg = KeySizeCheck_df[KeySizeCheck_df['Is Key Size'] == True].groupby(['Article','Colour'],as_index=False).agg({'SOH':'sum'})
    KeySizeCheck_keysize_agg = KeySizeCheck_keysize_agg.rename(columns={'SOH': 'Article Color Key Size SOH'})
    KeySizeCheck_df = pd.merge(KeySizeCheck_df, KeySizeCheck_keysize_agg[['Article', 'Colour', 'Article Color Key Size SOH']], on=['Article','Colour'], how='left')
    KeySizeCheck_df["Key Size Check 2"] = (KeySizeCheck_df["Article Color Key Size SOH"] >= Min_SOH_Per_KeySize_per_Article_Col).astype(int)

    # KeySizeCheck Step 3: Check if total number of key sizes for each Article-Color is above the minimum
    KeySizeCheck_df["Key Size Check 3"] = (KeySizeCheck_df["Article Color Num Key Sizes"] >= Min_Num_KeySize_per_Article_Col).astype(int)

    # KeySizeCheck Step 4: Check if total number of sizes with minimum SOH for each Article-Color is above the minimum
    KeySizeCheck_df['Min OH check'] = KeySizeCheck_df['SOH'] >= Min_SOH_Per_KeySize_per_Article_Col
    KeySizeCheck_nonkeysize_agg = KeySizeCheck_df[(KeySizeCheck_df['Min OH check'] == True)].groupby(['Article','Colour'],as_index=False).size()
    KeySizeCheck_df = pd.merge(KeySizeCheck_df, KeySizeCheck_nonkeysize_agg, on=['Article','Colour'], how='left')
    KeySizeCheck_df["Key Size Check 4"] = (KeySizeCheck_df["size"] >= Min_Num_Size_with_min_OH_per_Article_Col).astype(int)

    # Filter items that fail any of the previous 4 checks
    KeySizeCheck_filtered = KeySizeCheck_df[KeySizeCheck_df['Key Size Check 1'] + 
                                            KeySizeCheck_df['Key Size Check 2'] + 
                                            KeySizeCheck_df['Key Size Check 3'] +
                                            KeySizeCheck_df['Key Size Check 4']  == 4]

    # KeySizeCheck Step 5: Check if total number of colors for each Article is above the minimum
    KeySizeCheck_color_agg = KeySizeCheck_filtered.groupby(['Article'], as_index=False).agg({'Colour':'count'})
    KeySizeCheck_color_agg["Key Size Check"] = (KeySizeCheck_color_agg['Colour'] >= Min_Num_Colors_per_Article).astype(int)

    # Aggregate SOH by Article, and count number of colors
    soh_agg_df = soh_df.groupby(['Article'], as_index=False).agg({'SOH': 'sum', 'Colour':'count'})
    soh_agg_df['Key Size Check'] = (soh_agg_df['Article'].isin(KeySizeCheck_color_agg[KeySizeCheck_color_agg['Key Size Check'] == True]['Article']))
    processing_df = soh_agg_df.copy()
  
    # Reading from Item Master, to determine additional features, to be used to lookup the seasonal focus for that product
    params = {
        'host':'hgpost-sg-ees1zqm01001-ap-southeast-1.hologres.aliyuncs.com',
        'port': 80, 
        'dbname':'data_center',
        'user':os.getenv('HOLOGRES_USER'),
        'password':os.getenv('HOLOGRES_PASS'),
    }

    conn = psycopg2.connect(**params)
    query = "select article, item_class_name_desc, item_sub_class_desc, item_category_desc, sku_theme, launch_desc, item_group_desc, sku_size from dim.item_master where brand='CK' AND sales_org='3000' AND (label_id='CKSL' OR label_id='CKCK')"
    item_master_df = pd.read_sql(query, conn)
    conn.close()
    item_master_df = item_master_df.rename(columns={'article': 'Article', 'item_class_name_desc': 'Class', 
                                                    'item_sub_class_desc':'Sub Class', 'item_category_desc':'Category Name',
                                                    'sku_theme':'Theme', 'launch_desc':'Season', 
                                                    'item_group_desc': 'Category','sku_size':'Size'})
    
    # Getting the info for each product
    item_master_df = item_master_df.groupby(['Article'], as_index=False).agg({'Class': 'first', 'Sub Class':'first', 'Category Name':'first',
                                                                              'Theme':'first', 'Season':'first', 'Category':'first', 'Size':'first'})
    processing_df = processing_df.merge(item_master_df, on = "Article")

    # The shoes are either boots (determined by 'Class'), or in one of the types in shoe_types (determined by 'SubClass')
    shoe_types = list(calendar_df.columns)  # Getting seasonal focus info
    processing_df['Seasonal Focus'] = processing_df.apply(lambda row: calendar_df[row['Class']][week]
                                                          if row['Class'] == 'Boots'
                                                          else (calendar_df[row['Sub Class']][week] if row['Sub Class'] in shoe_types else ''),
                                                          axis=1)
                    
    # Reading Marketing Push info from worksheet
    marketing_df = params_dict["Marketing Push"].iloc[:, 0:5]
    marketing_items = marketing_df[(marketing_df['Year']== year) & (marketing_df['Week']== week)]

    # Appending marketing list onto the Rough Working List
    processing_df['Marketing'] = processing_df['Article'].isin(marketing_items['Article'])

    # Reading Family Mapping file for each product category
    FamilyGrouping_ref = params_dict["DAT13 Core List Tracker (3)"].iloc[:, 1:3]
    FamilyGrouping_ref.rename(columns = {'Initial Article': 'Family Mapping', 'New Article': 'Article'}, inplace=True)

    articles_to_add = list(set(soh_df[~soh_df['Article'].isin(FamilyGrouping_ref['Article'].to_list())]['Article']))
    list_base_articles = list(['-'.join((article.split('_')[0]).split('-')[:2]) for article in articles_to_add])
    grouping_base_articles = pd.DataFrame({'Family Mapping': list_base_articles, 'Article': articles_to_add})
    
    FamilyGrouping_ref = pd.concat([FamilyGrouping_ref, grouping_base_articles], axis=0, ignore_index=True)

    # Appending family grouping onto the Rough Working List
    processing_df = processing_df.merge(FamilyGrouping_ref, on = "Article", how='left')
    
    # Reading discount info from datatable
    conn = psycopg2.connect(**params)

    # For countries without e-commerce sales, use sales datatable
    if country in ['SAUDI ARABIA', 'VIETNAM', 'THAILAND', 'INDONESIA', 'INDIA']:
        # Use ecom sales datatable for sf2 countries
        query = "SELECT distinct t1.article as article, \
            (t1.total_discount_amount_on_item / NULLIF(t1.total_net_price, 0)) * 100 AS markdown_percent, \
            t3.min_article_launch_yr, t3.min_article_launch_wk \
            FROM ads.ads_ecom_sf2_sales_hi as t1 \
            INNER JOIN ( \
                SELECT article, country, MAX(DATE(order_ts)) as max_date from ads.ads_ecom_sf2_sales_hi \
                GROUP BY article, country \
            ) AS t2 \
            ON t1.article = t2.article and t1.country = t2.country and DATE(t1.order_ts) = t2.max_date \
            INNER JOIN( \
                SELECT article, country, MIN(article_launch_yr) AS min_article_launch_yr, MIN(article_launch_wk) AS min_article_launch_wk \
                FROM ads.ads_ckg_stock_type_di \
                GROUP BY article, country \
            ) AS t3 \
            ON t2.article = t3.article and t2.country = t3.country \
            WHERE t1.country = %s"

    else: 
        # Use ecom sales datatable for all other countries
        query = "SELECT distinct t1.articleno as article, t1.discount_condition, \
                (t1.discount_amount / NULLIF(t1.total_net_price, 0)) * 100 AS markdown_percent, \
                t3.min_article_launch_yr, t3.min_article_launch_wk\
                FROM ads.ads_ckg_ecom_salesfact as t1\
                INNER JOIN (\
                    SELECT articleno, country, MAX(date) as max_date from ads.ads_ckg_ecom_salesfact\
                    GROUP BY articleno, country\
                ) AS t2\
                ON t1.articleno = t2.articleno and t1.country = t2.country and t1.date = t2.max_date\
                INNER JOIN(\
                    SELECT article, country, MIN(article_launch_yr) AS min_article_launch_yr, MIN(article_launch_wk) AS min_article_launch_wk\
                    FROM ads.ads_ckg_stock_type_di\
                    GROUP BY article, country\
                ) AS t3\
                ON t2.articleno = t3.article and t2.country = t3.country\
                WHERE shop_no = 'CK.COM' and t1.country = %s"
       
    discount_df = pd.read_sql(query, conn, params=[country]).dropna().copy()
    discount_df_min_launchdates = discount_df.copy()

    # Get min launch date for each article
    discount_df_min_launchdates['article_launch_wk'] = pd.to_timedelta(discount_df_min_launchdates['min_article_launch_wk'] * 7, unit='d')
    discount_df_min_launchdates['min_date'] = pd.to_datetime(discount_df_min_launchdates['min_article_launch_yr'].astype(int).astype(str), format='%Y') + discount_df_min_launchdates['article_launch_wk']
    min_launch_dates = discount_df_min_launchdates.groupby('article')['min_date'].min().reset_index()

    # Merge min launch date onto discount df
    discount_df.drop(columns=['min_article_launch_yr', 'min_article_launch_wk'], inplace=True)
    discount_df = discount_df.merge(min_launch_dates, on='article', how='left')

    # Get discount status for each article, there is different discount country
    discount_df.rename(columns={'min_date':'Launch Date',
                                'article':'Article', 
                                'discount_status':'Latest MD Status',
                                'markdown_percent': 'Discount Percentage'},inplace=True)

    # Calculating number of weeks launched for products
    discount_df["Weeks Launched"] = discount_df['Launch Date'].apply(lambda x: np.ceil(np.timedelta64(pd.to_datetime('today') - pd.to_datetime(x),'D').astype(int)/7))

    # If discount percentage is great than 70%, then these are likely to be giveaways and not actual markdowns
    discount_df['Latest MD Status'] = discount_df['Discount Percentage'].apply(lambda x: 'Regular' if x > 0.7 or x < 0.1 else 'Markdown')
    discount_df.drop('Discount Percentage', axis = 1, inplace = True)
    
    # Appending launch weeks onto the Rough Working List
    processing_df = processing_df.merge(discount_df, on = "Article", how='left')

    # Reading New Arrivals info from worksheet
    new_arrivals = params_dict["New Arrivals"].iloc[0:, 0:2]
    new_arrivals.columns = ['Week', 'Article']
    new_arrivals = new_arrivals[new_arrivals['Week'] >= week-2]     # takes last 3 weeks of new arrivals
    processing_df['New Arrival'] = processing_df['Article'].isin(new_arrivals['Article'])

    # Reading Stock Type info from stock type weekly datatable, to determine which products are seasonal
    conn = psycopg2.connect(**params)
    query = "select article, stock_type from ads.ads_ckg_stock_type_wi where country = %s AND brand = 'CK'"
    stocktype_ref = pd.read_sql(query, conn, params=[country])
    stocktype_ref.rename(columns={'article':'Article', 'stock_type':'Stock Type'},inplace=True)
    stocktype_ref_dict = {'SEASONAL':'3 SEASONAL',
                          'REPEATS':'2 REPEATS',
                          'TERMINAL': '1 TERMINAL',
                          'OBSOLETE': '0 OBSOLETE',
                          }
    stocktype_ref['Stock Type'] = stocktype_ref['Stock Type'].map(stocktype_ref_dict)

    # Appending stock type onto the Rough Working List
    processing_df = processing_df.merge(stocktype_ref, on = "Article", how = 'left')
    processing_df = processing_df.drop_duplicates(subset=['Article'])

    # Reading Repeats info from worksheet
    repeats = params_dict["Repeats"].iloc[0:, 0:2]
    repeats.columns = ['Week', 'Article']
    repeats = repeats[repeats['Week'] >= week-2]
    processing_df['Repeat'] = processing_df['Article'].isin(repeats['Article'])

    # For Japan, do not remove markdowns, special request from them
    if country != 'JAPAN':
        processing_df = processing_df[processing_df['Latest MD Status'] != 'Markdown'].copy()

    # Reading Product ID, Category ID, and Stage ID from worksheets
    ref_sheets = {
    "PID_ref": {"sheet_name": "countryappend", "columns": slice(0,2), "names": ["Country", "PID"]},
    "CID_ref": {"sheet_name": "countryappend", "columns": slice(3,5), "names": ["Country", "CID"]},
    "SID_ref": {"sheet_name": "stageid", "columns": slice(0,2), "names": ["Class", "SID"]}
    }

    # Get list of seasonal articles
    seasonal_article_tuple = tuple(processing_df[processing_df['Stock Type']=='3 SEASONAL']['Article'])

    conn = psycopg2.connect(**params)

    # Use ecom sales datatable for all other countries
    query = f'''
    SELECT articleno as article, SUM(soldqty) as total_sold_qty_seasonal
    FROM ads.ads_ckg_ecom_salesfact 
    WHERE articleno IN {seasonal_article_tuple}
    GROUP BY articleno
    ORDER BY total_sold_qty_seasonal DESC
    '''
        
    seasonal_sales = pd.read_sql(query, conn).dropna().copy()
    conn.close()

    seasonal_sales.columns = seasonal_sales.columns.str.capitalize()

    conn = psycopg2.connect(**params)

    # Use ecom sales datatable for all other countries
    query = f'''SELECT articleno as article, SUM(soldqty) as total_sold_qty_weekly
            FROM ads.ads_ckg_ecom_salesfact 
            WHERE (articleno SIMILAR TO 'CK[0-9]%' OR articleno SIMILAR TO 'SL[0-9]%')
            AND date >= '{start_of_week}'
            GROUP BY articleno
            ORDER BY total_sold_qty_weekly DESC
            '''
        
    weekly_sales = pd.read_sql(query, conn).dropna().copy()
    conn.close()

    # Capitalize the column names
    weekly_sales.columns = weekly_sales.columns.str.capitalize()

    conn = psycopg2.connect(**params)

    # Use ecom sales datatable for all other countries
    query = f'''
    SELECT DISTINCT article, sku_product_name FROM dim.item_master 
    WHERE (sku_product_name = 'GABINE' OR sku_product_name = 'KOA' OR sku_product_name = 'CHARLOT' OR sku_product_name = 'PERLINE' OR sku_product_name = 'PETRA' OR sku_product_name = 'TONI')
    '''
        
    product_names = pd.read_sql(query, conn).dropna().copy()
    conn.close()

    gabine_articles = product_names[product_names['sku_product_name']=='GABINE']['article'].to_list()
    koa_articles = product_names[product_names['sku_product_name']=='KOA']['article'].to_list()
    perline_articles = product_names[product_names['sku_product_name']=='PERLINE']['article'].to_list()
    charlot_articles = product_names[product_names['sku_product_name']=='CHARLOT']['article'].to_list()
    petra_articles = product_names[product_names['sku_product_name']=='PETRA']['article'].to_list()
    toni_articles = product_names[product_names['sku_product_name']=='TONI']['article'].to_list()

    # The sheets are saved into a dictionary
    ref_dfs = {}
    for ref_name, ref_info in ref_sheets.items():
        df = params_dict[ref_info["sheet_name"]].iloc[0:, ref_info["columns"]]
        df.columns = ref_info['names']
        ref_dfs[ref_name] = df

    # Function for initial sorting of products by category
    def product_sorting(product_category):
        if product_category == 'Bags':
            article_prefix = "CK2|CK6|SL2|SL6|CK11"
        elif product_category == 'Shoes':
            article_prefix = "CK1|CK9|SL1"
        elif product_category == 'Sunglasses':
            article_prefix = "CK3"
        elif product_category == 'Jewellery':
            article_prefix = "CK5"
        elif product_category == 'Accessories':
            article_prefix = "SL12|CK8"

        # Filtering for products by category
        selected_category = processing_df[processing_df['Article'].str.contains(article_prefix)]

        # Whether articles are in these special categories
        selected_category['Gabine'] =  selected_category['Article'].isin(gabine_articles)
        selected_category['Koa'] =  selected_category['Article'].isin(koa_articles)
        selected_category['Charlot'] =  selected_category['Article'].isin(charlot_articles)
        selected_category['Perline'] =  selected_category['Article'].isin(perline_articles)
        selected_category['Petra'] =  selected_category['Article'].isin(petra_articles)
        selected_category['Toni'] =  selected_category['Article'].isin(toni_articles)

        # Get best sellers by week and by season
        selected_category_copy = selected_category.copy()
        selected_category_seasonal_best = pd.merge(selected_category_copy, seasonal_sales, on='Article', how='left').sort_values(['Total_sold_qty_seasonal'], ascending=False).head(10)
        selected_category['Best Seller Seasonal'] =  selected_category['Article'].isin(selected_category_seasonal_best['Article'])
        selected_category_weekly_best = pd.merge(selected_category_copy, weekly_sales, on='Article', how='left').sort_values(['Total_sold_qty_weekly'], ascending=False).head(10)
        selected_category['Best Seller Weekly'] =  selected_category['Article'].isin(selected_category_weekly_best['Article'])

        # Getting minimum weeks launched for each family map grouping 
        GroupMinWeeks_df = selected_category.groupby(['Family Mapping'],as_index=False).agg({'Weeks Launched': 'min'})
        GroupMinWeeks_df.rename(columns={'Weeks Launched':'Group Min Weeks'},inplace=True)
        products_sorting = selected_category.merge(GroupMinWeeks_df, on = "Family Mapping", how='left')

        # Summing SOH by Theme for sorting purposes later
        SOHbyTheme_df = selected_category.groupby(['Theme'],as_index=False).agg({'SOH': 'sum'})
        SOHbyTheme_df.rename(columns={'SOH':'SOH By Theme'},inplace=True)
        SOHbyTheme_df.dropna(subset=['Theme'], inplace=True)
        products_sorting = selected_category.merge(SOHbyTheme_df, on = "Theme", how='left')

        # Add in products in Core Group (only for Bags and Shoes)
        if product_category in ('Bags', 'Shoes'):
            core_ref = params_dict[f"{product_category}CORE"].iloc[:, 0:2]
            core_ref.columns = ["Article", "Core Group"]
            core_ref = core_ref.groupby('Article').agg('first').reset_index(names='Article')
            products_sorting = products_sorting.merge(core_ref, on = "Article", how = "left")

        # Initial sorting using the columns added thus far
        sorted_products = products_sorting.sort_values(['Marketing', 'New Arrival', 'Gabine', 'Koa', 
                                                        'Best Seller Seasonal', 'Best Seller Weekly', 
                                                        'Charlot', 'Perline', 'Petra', 'Toni', 
                                                        'Repeat', 'Core Group', 'Family Mapping', 'Key Size Check', 
                                                        'Colour', 'Weeks Launched','Stock Type','SOH By Theme', 'SOH'], 
                                                        ascending=[False, False, False, False,
                                                                False, False, 
                                                                False, False, False, False,
                                                                False, False, False, False,
                                                                False, True, False, False, False])
        
        # sorted_products = sorted_products.reindex(columns=['Article', 'Marketing', 'New Arrival', 'Gabine', 'Koa', 
        #                                                     'Best Seller Seasonal', 'Best Seller Weekly', 
        #                                                     'Charlot', 'Perline', 'Petra', 'Toni',  
        #                                                     'Repeat', 
        #                                                     'Core Group', 'Family Mapping', 'Key Size Check',
        #                                                     'Colour','Weeks Launched', 'Stock Type', 'Seasonal Focus',
        #                                                     'SOH', 'Class', 'Sub Class', 'Theme', 'Season', 'Category',
        #                                                     'Category Name', 'Size', 'First Sales Date', 'SOH By Theme'])
        
        output = sorted_products.copy()

        # Sorting and putting core + family mapping together
        if product_category in ('Bags', 'Shoes'):
            output.replace('NaN', np.nan, inplace=True)
            output['Core_pos'] = output['Core Group'].fillna(0)  # Create a NaN_pos column to indicate not in core group
            output['Map_pos'] = output['Family Mapping'].fillna(0) # Create a Map_pos column to indicate no family mapping
            output = output.reset_index(drop=True)

        # Adding in Product ID, Catalog ID, and Values columns
        output['Country'] = country
        output = output.merge(ref_dfs['PID_ref'], on = "Country", how = "left")
        output = output.merge(ref_dfs['CID_ref'], on = "Country", how = "left")
        output = output.merge(ref_dfs['SID_ref'], on = "Class", how = "left") 
        if product_category == 'Shoes':
                output = output.merge(ref_dfs['SID_ref'], left_on = "Category Name", right_on = "Class", how = "left") #this creates SID_x and SID_y
                output['Class'] = output['Class_x']
                output.drop(columns=['Class_x'], inplace=True)

        output["Product ID"] = output["Article"].astype(str) +"-"+ output["PID"]
        output["Catalog ID"] = "storefront_ck-" + output["CID"]
        output['Values'] = 'bottom'
        
        return output
    
    # Function to group products by core group and family mapping
    def product_grouping(sorted_items):

        # Sort by columns
        sort_columns = ['Marketing', 'New Arrival', 'Repeat', 'Family Mapping',
                        'Key Size Check', 'Colour', 'Weeks Launched', 'Stock Type', 
                        'SOH By Theme', 'SOH']
        sort_order = [False, False, False, True,
                        False, False, True, False, 
                        False, False]
        sorted_items_core = sorted_items.sort_values(sort_columns, ascending=sort_order).copy()

        # Remove duplicates
        sorted_items_core.dropna(subset=['Article'], inplace=True)
        sorted_by_groups = sorted_items_core.drop_duplicates(subset=['Article'], keep='first').copy()
        sorted_by_groups.reset_index(drop=True, inplace=True)

        # Put low priority items at the bottom
        conditions = sorted_by_groups.index[(sorted_by_groups['Marketing'] == False) & 
                                            (sorted_by_groups['New Arrival'] == False) & 
                                            (sorted_by_groups['Repeat'] == False) & 
                                            (sorted_by_groups['Core Group'].isna())]

        no_priority = sorted_by_groups.iloc[conditions]
        priority = sorted_by_groups.drop(index=conditions)
        sorted_by_groups = pd.concat([priority, no_priority]).reset_index(drop=True)

        return sorted_by_groups

    # Bags
    sorted_bags_core = product_sorting('Bags')
    sorted_bags_by_groups = product_grouping(sorted_bags_core)

    # Prepping MC, NA and SC IDs for copy and paste process afterwards
    SFupload_df = sorted_bags_by_groups.copy()

    def add_mc_tags(s):
        if s['Category'] in ['Small Leather Goods', 'Wallet']:
            if s['Class'] in ['Mini Bag']:
                return "mcbags"
            return "mcwallets"
        elif s['Category'] == 'Bags':
            return "mcbags"
        else:
            return 0
        
    def add_na_tags(s):
        if s['Is New Arrival'] == 1:
            if s['Category'] == 'Small Leather Goods':
                return "newarrivals-mcwallets"
            elif s['Category'] == 'Bags':
                return "newarrivals-mcbags"
        return 0

    def add_slgtags(s):
        if s['Class'] == 'Wallet':
            if s['Size'] == 'Extra Small':
                return 'longwallets'
            elif s['Size'] == 'Extra Extra Small':
                return 'shortwallets'
        elif s['Class'] in ['Phone Pouch', 'Pouch', 'Wristlet']:
            return 'wristletsandpouches'
        elif s['Class'] in ['Mini Bag', 'Mini Purse']:
            return 'minibags'
        

    # Main Category
    SFupload_df['Is Main Category'] = ((SFupload_df['Category'] == 'Small Leather Goods') | (SFupload_df['Category'] == 'Wallet') | (SFupload_df['Category'] == 'Bags')).astype(int)
    SFupload_df['Prep MC ID'] = SFupload_df.apply(add_mc_tags, axis=1)
    SFupload_df['MC ID'] = np.where(SFupload_df['Is Main Category'] == 1,
                                    SFupload_df['Prep MC ID'],
                                    SFupload_df['SID'])

    # New Arrival
    SFupload_df['Is New Arrival'] = SFupload_df['New Arrival'].astype(int)
    SFupload_df['Prep NA ID'] = SFupload_df.apply(add_na_tags, axis=1)
    SFupload_df['NA ID'] = np.where(SFupload_df['Is New Arrival'] == 1,
                                    SFupload_df['Prep NA ID'],
                                    SFupload_df['MC ID'])
    SFupload_df['NA ID'] = np.where(SFupload_df['NA ID'] == 0, 
                                    SFupload_df['MC ID'],
                                    SFupload_df['NA ID'])
    # Stage ID
    SFupload_df['Stage ID Check'] = ((SFupload_df['Marketing'] == False) & 
                                        (SFupload_df['New Arrival'] == False) & 
                                        (SFupload_df['Repeat'] == False)) | (SFupload_df['Is New Arrival'] == 1) | (SFupload_df['Is Main Category'] == 1).astype(int)
    SFupload_df['SC ID'] = np.where(SFupload_df['Stage ID Check'] == 1, SFupload_df['SID'], 0)
    SFupload_df['SC ID'] = np.where((SFupload_df['SC ID'].isnull()) | (SFupload_df['SC ID'].str.len() == 0),
                                    SFupload_df.apply(add_slgtags, axis=1), 
                                    SFupload_df['SC ID'])

    def get_final_output(product_category_string):
        # Filter for MC ID
        print("-------------------------------------------------------------------------------")
        mcid_one = SFupload_df.loc[SFupload_df['Is Main Category'] == 1]
        mcid_one = mcid_one[['Product ID', 'MC ID', 'Catalog ID', 'Values']]
        mcid_one.rename(columns={'MC ID': 'Category ID'}, inplace=True)

        print(f'Output for MC ID {product_category_string} prepared!')
        print("-------------------------------------------------------------------------------")

        # Filter for SC ID
        if product_category_string == 'Shoes':  # For shoes, there is SC ID 1 and SC ID 2
            scid_one = SFupload_df.loc[SFupload_df['Stage ID Check'] == 1]
            scid_one = scid_one[['Product ID', 'SC ID 1', 'Catalog ID', 'Values']]
            scid_one.rename(columns={'SC ID 1': 'Category ID'}, inplace=True)

            scid_two = SFupload_df.loc[SFupload_df['Stage ID Check'] == 1]
            scid_two = scid_two[['Product ID', 'SC ID 2', 'Catalog ID', 'Values']]
            scid_two.rename(columns={'SC ID 2': 'Category ID'}, inplace=True)

            print(f'Output for SC ID 1 and SC ID 2 {product_category_string} prepared!')

        else:  # For others, there is only SC ID
            scid_one = SFupload_df.loc[SFupload_df['Stage ID Check'] == 1]
            scid_one = scid_one[['Product ID', 'SC ID', 'Catalog ID', 'Values']]
            scid_one.rename(columns={'SC ID': 'Category ID'}, inplace=True)
            
            scid_two = pd.DataFrame()  # Empty dataframe as a placeholder

            print(f'Output for SC ID {product_category_string} prepared!')
        print("-------------------------------------------------------------------------------")

        # Filter for NA ID
        naid_one = SFupload_df.loc[SFupload_df['Is New Arrival'] == 1]
        naid_one = naid_one[['Product ID', 'NA ID', 'Catalog ID', 'Values']]
        naid_one.rename(columns={'NA ID': 'Category ID'}, inplace=True)

        print(f'Output for NA ID {product_category_string} prepared!')
        print("-------------------------------------------------------------------------------")

        # Combine all dataframes into one dataframe
        all_tagged = pd.concat([mcid_one, scid_one, scid_two, naid_one], ignore_index=True)

        if product_category_string == 'Shoes':
            ck17_article_df = all_tagged[all_tagged['Product ID'].str.startswith('CK17')]  # Remove CK17
            blank_category_df = all_tagged[all_tagged['Category ID'].isnull()]  # Remove those without Category ID
            filtered_df = pd.concat([ck17_article_df, blank_category_df])
            all_tagged.drop(filtered_df.index, inplace=True)

        # Instead of saving, return the dataframes
        print(f'Final Output for {product_category_string} for this week prepared!')
        
        return all_tagged, naid_one
    
    bags_sorted_df, naid_one_bagsslg = get_final_output('Bags')

    # Shoes
    sorted_shoes_core  = product_sorting('Shoes')
    sorted_shoes_by_groups = product_grouping(sorted_shoes_core)

    # Prepping MC, NA and SC IDs for copy and paste process afterwards
    SFupload_df = sorted_shoes_by_groups.copy()

    # Add MC ID column based on conditions
    SFupload_df['Is Main Category'] = ((SFupload_df['Category'] == 'Footwear') | (SFupload_df['Category'] == 'Kids Footwear')).astype(int)
    SFupload_df['Prep MC ID'] = SFupload_df['Is Main Category'].apply(lambda x: "mcshoes" if x == 1 else 0)
    SFupload_df['MC ID'] = np.where(SFupload_df['Is Main Category'] == 1, 
                                    SFupload_df['Prep MC ID'], 
                                    SFupload_df['SID_x'])

    # Add NA ID column based on conditions
    SFupload_df['Is New Arrival'] = SFupload_df['New Arrival'].astype(int)
    SFupload_df['Prep NA ID'] = SFupload_df['Is New Arrival'].apply(lambda x: "newarrivals-mcshoes" if x == 1 else 0)
    SFupload_df['NA ID'] = np.where(SFupload_df['Is New Arrival'] == 1,
                                    SFupload_df['Prep NA ID'],
                                    SFupload_df['MC ID'])

    # Add SC ID column based on conditions
    SFupload_df['Stage ID Check'] = ((SFupload_df['Marketing'] == False) & 
                                        (SFupload_df['New Arrival'] == False) & 
                                        (SFupload_df['Repeat'] == False)) | (SFupload_df['Is New Arrival'] == 1) | (SFupload_df['Is Main Category'] == 1).astype(int)
    SFupload_df['SC ID 1'] = np.where(SFupload_df['Stage ID Check'] == 1,
                                        SFupload_df['SID_x'],
                                        0)
    SFupload_df['SC ID 2'] = np.where(SFupload_df['Stage ID Check'] == 1,
                                        SFupload_df['SID_y'],
                                        0)

    # Save to working output folder
    shoes_sorted_df, naid_one_shoes = get_final_output('Shoes')

    # Sunglasses
    sorted_sg = product_sorting('Sunglasses')
    SFupload_df = sorted_sg.copy()

    # Add MC ID column based on conditions
    SFupload_df['Is Main Category'] = (SFupload_df['Category'] == 'Sunglasses').astype(int)
    SFupload_df['Prep MC ID'] = SFupload_df['Is Main Category'].apply(lambda x: "mcsunglasses" if x == 1 else 0)
    SFupload_df['MC ID'] = np.where(SFupload_df['Is Main Category'] == 1,
                                    SFupload_df['Prep MC ID'],
                                    SFupload_df['SID'])

    # Add NA ID column based on conditions
    SFupload_df['Is New Arrival'] = SFupload_df['New Arrival'].astype(int)
    SFupload_df['Prep NA ID'] = SFupload_df['Is New Arrival'].apply(lambda x: "newarrivals-mcsunglasses" if x == 1 else 0)
    SFupload_df['NA ID'] = np.where(SFupload_df['Is New Arrival'] == 1,
                                    SFupload_df['Prep NA ID'],
                                    SFupload_df['MC ID'])

    # Add SC ID column based on conditions
    SFupload_df['Stage ID Check'] = ((SFupload_df['Marketing'] == False) & 
                                        (SFupload_df['New Arrival'] == False) & 
                                        (SFupload_df['Repeat'] == False)) | (SFupload_df['Is New Arrival'] == 1) | (SFupload_df['Is Main Category'] == 1).astype(int)
    SFupload_df['SC ID'] = np.where(SFupload_df['Stage ID Check'] == 1,
                                    "mcsunglasses",
                                    "0")

    # Save to working output folder
    sg_sorted_df, naid_one_sg  = get_final_output('Sunglasses')

    # Jewellery
    sorted_j = product_sorting('Jewellery')
    SFupload_df = sorted_j.copy()


    # Add MC ID column based on conditions
    SFupload_df['Is Main Category'] = (SFupload_df['Category'].isin(['Bracelet', 'Earring', 'Necklaces', 'Ring', 'Charm'])).astype(int)
    SFupload_df['Prep MC ID'] = SFupload_df['Is Main Category'].apply(lambda x: "mcjewellery" if x == 1 else 0)
    SFupload_df['MC ID'] = np.where(SFupload_df['Is Main Category'] == 1,
                                    SFupload_df['Prep MC ID'],
                                    SFupload_df['SID'])

    # Add NA ID column based on conditions
    SFupload_df['Is New Arrival'] = SFupload_df['New Arrival'].astype(int)
    SFupload_df['Prep NA ID'] = SFupload_df['Is New Arrival'].apply(lambda x: "newarrivals-mcjewellery" if x == 1 else 0)
    SFupload_df['NA ID'] = np.where(SFupload_df['Is New Arrival'] == 1,
                                    SFupload_df['Prep NA ID'],
                                    SFupload_df['MC ID'])

    # Add SC ID column based on conditions
    SFupload_df['Stage ID Check'] = ((~SFupload_df['Marketing']) & 
                                        (~SFupload_df['New Arrival']) & 
                                        (~SFupload_df['Repeat'])) | (SFupload_df['Is New Arrival'] == 1) | (SFupload_df['Is Main Category'] == 1).astype(int)
    SFupload_df['SC ID'] = np.where(SFupload_df['Stage ID Check'] == 1,
                                    "mcjewellery",
                                    "0")

    # Save to working output folder
    j_sorted_df, naid_one_j = get_final_output('Jewellery')

    # Accessories
    sorted_acc = product_sorting('Accessories')
    SFupload_df = sorted_acc.copy()

    # Add MC ID column based on conditions
    SFupload_df['Is Main Category'] = SFupload_df['Category'].isin(['Bracelet', 'Earring', 'Necklaces', 'Ring', 'Charm']).astype(int)
    SFupload_df['Prep MC ID'] = SFupload_df['Is Main Category'].apply(lambda x: "mcaccessories" if x == 1 else 0)
    SFupload_df['MC ID'] = np.where(SFupload_df['Is Main Category'] == 1,
                                    SFupload_df['Prep MC ID'],
                                    SFupload_df['SID'])

    # Add NA ID column based on conditions
    SFupload_df['Is New Arrival'] = SFupload_df['New Arrival'].astype(int)
    SFupload_df['Prep NA ID'] = SFupload_df['Is New Arrival'].apply(lambda x: "newarrivals-mcaccessories" if x == 1 else 0)
    SFupload_df['NA ID'] = np.where(SFupload_df['Is New Arrival'] == 1,
                                    SFupload_df['Prep NA ID'],
                                    SFupload_df['MC ID'])

    # Add SC ID column based on conditions
    SFupload_df['Stage ID Check'] = ((~SFupload_df['Marketing']) & 
                                        (~SFupload_df['New Arrival']) & 
                                        (~SFupload_df['Repeat'])) | (SFupload_df['Is New Arrival'] == 1) | (SFupload_df['Is Main Category'] == 1).astype(int)
    SFupload_df['SC ID'] = np.where(SFupload_df['Stage ID Check'] == 1,
                                    "mcaccessories",
                                    "0")

    # Save to working output folder
    acc_sorted_df, naid_one_acc  = get_final_output('Accessories')

    # Gift Set (manual entry of data)
    giftset_data = {
        'Article': ['CK17-70840500', 'CK17-50681039', 'CK17-70701232', 'CK17-50681040', 'CK17-50681039'],
        'Category ID': ['mcbags', 'mcbags', 'mcbags', 'mcbags', 'mcbags'],
        'Values': ['bottom', 'bottom', 'bottom', 'bottom', 'bottom']
    }

    # Construct giftset dataframe
    giftset_data_df = pd.DataFrame(giftset_data)
    giftset_data_df['Country'] = country
    gs_sorted_df = giftset_data_df.merge(ref_dfs['PID_ref'], on = "Country", how = "left")
    gs_sorted_df = gs_sorted_df.merge(ref_dfs['CID_ref'], on = "Country", how = "left")
    gs_sorted_df["Product ID"] = gs_sorted_df["Article"].astype(str) +"-"+ gs_sorted_df["PID"]
    gs_sorted_df["Catalog ID"] = "storefront_ck"+ "-" + gs_sorted_df["CID"]

    # Merge SOH info with giftset dataframe
    soh_gs_df = soh_df.copy()
    gs_sorted_df = gs_sorted_df.merge(soh_gs_df[['Article', 'SOH']], on='Article', how='left')
    gs_sorted_df = gs_sorted_df[gs_sorted_df['SOH'] > 0]
    gs_sorted_df.drop(["Article", "Country", "PID", "CID"], axis=1, inplace=True)
    gs_sorted_df = gs_sorted_df[["Product ID", "Category ID", "Catalog ID", "Values"]]

    # Combine all new arrivals across the categories
    new_arrivals_combined = pd.DataFrame(pd.concat([naid_one_shoes, naid_one_bagsslg, naid_one_sg, naid_one_j, naid_one_acc], 
                                                    axis=0, ignore_index=True))
    new_arrivals_combined = new_arrivals_combined.sort_values(['Category ID'], ascending=[True])

    # Add Family Mapping column
    all_sorted_categories = pd.concat([sorted_bags_by_groups, sorted_shoes_by_groups, sorted_sg, sorted_j, sorted_acc])
    new_arrivals_combined = pd.merge(new_arrivals_combined, all_sorted_categories[['Product ID', 'Family Mapping']], on = 'Product ID', how = 'left')

    # Add Map Order column to keep track of sequence of family mapping
    sequence_dict = {}

    for num, row in new_arrivals_combined.iterrows():
        family_mapping = row['Family Mapping']
        category_id = row['Category ID']
        
        if pd.notna(family_mapping):
            if (family_mapping, category_id) not in sequence_dict:
                sequence_dict[(family_mapping, category_id)] = num
            new_arrivals_combined.loc[num, 'Map Order'] = sequence_dict[(family_mapping, category_id)]
        else:
            new_arrivals_combined.loc[num, 'Map Order'] = num
        
    new_arrivals_combined.sort_values('Map Order', inplace=True)

    # Add Cat Rank column using Category ID, rank is based on the number of unique categories
    cat_ranks = {category: rank + 1 for rank, category in enumerate(new_arrivals_combined['Category ID'].unique())}
    new_arrivals_combined['Cat Rank'] = new_arrivals_combined['Category ID'].map(cat_ranks)

    # Giving an individual category rank to each product within the same category
    new_arrivals_combined['Individual Cat Rank'] = new_arrivals_combined.groupby('Category ID').cumcount() + 1

    # Add MOD Rank column        
    def mod_rank(row):
        # Values for each category are pre-defined, the greater the number, 
        # the more the products are divided up into smaller groups when sorting into a sequence 
        category_modulos = {
            'newarrivals-mcshoes': 4,
            'newarrivals-mcbags': 3,
            'newarrivals-mcwallets': 2,
            'newarrivals-mcsunglasses': 2,
            'newarrivals-mcjewellery': 2,
            'newarrivals-mcaccessories': 2
        }
        category_id = row['Category ID']
        individual_cat_rank = row['Individual Cat Rank']

        if category_id in category_modulos:
            return individual_cat_rank % category_modulos[category_id]
        else:
            return 0

    new_arrivals_combined['MOD Rank'] = new_arrivals_combined.apply(mod_rank, axis=1)

    # Add Sequence column
    last_category_id = None
    family_map_sequence = {}
    seq_counter = 0
    seq_list = []

    # Loop through each row in new_arrivals_combined dataframe to add sequence number
    for _, row in new_arrivals_combined.iterrows():

        # If row has a different category to the previous row, reset seq_counter to 1
        if row['Category ID'] != last_category_id:
            seq_counter = 1
            last_category_id = row['Category ID']

        # If row has same category to the previous row, but MOD rank goes back to 1 meaning the start of a new sequence, increase seq_counter
        # Mod ranking will be 1,2,3,...,0
        elif row['MOD Rank'] == 1:
            if row['Family Mapping'] not in family_map_sequence:
                seq_counter += 1
            else:
                seq_list.append(family_map_sequence[row['Family Mapping']])
                seq_counter += 1
                continue

        family_map_sequence[row['Family Mapping']] = seq_counter
        seq_list.append(seq_counter)

    new_arrivals_combined['Sequence'] = seq_list    

    # Sort by Sequence, Individual Cat Rank, Cat Rank, Mod Rank
    new_arrivals_combined_sorted = new_arrivals_combined.sort_values(['Sequence', 'Cat Rank', 'Individual Cat Rank', 
                                                                      'MOD Rank', 'Values', 'Catalog ID', 'Category ID'], 
                                                                      ascending=[True, True, True, 
                                                                                 False, False, False, False])
                                                                
    new_arrivals_combined_sorted = new_arrivals_combined_sorted.drop(columns=['Cat Rank', 'Individual Cat Rank', 'MOD Rank', 
                                                                              'Sequence', 'Family Mapping', 'Map Order'])
    new_arrivals_combined_sorted['Category ID'] = 'newarrivals'
    new_arrivals_combined_sorted.reset_index(drop=True, inplace=True)

    # All sorted dataframes
    all_categories_sorted = pd.concat([gs_sorted_df, shoes_sorted_df, bags_sorted_df, 
                                        sg_sorted_df, j_sorted_df, acc_sorted_df], axis=0, ignore_index=True)

    # Add New Arrivals to sorted dataframes, final output will have new arrivals as separate section
    final_sorted = pd.concat([all_categories_sorted, new_arrivals_combined_sorted], axis=0, ignore_index=True)

    # Lookup Product ID using country
    split_string = "-" + ref_dfs['PID_ref'].loc[ref_dfs['PID_ref']["Country"] == country, "PID"].values[0]

    # Add Product ID to Article as suffix
    final_sorted['Article'] = final_sorted['Product ID'].str.split(split_string).str[0]
    final_sorted['Photo'] = ''

    # Reading Marketing Push from worksheet, to retrieve products with specified Category IDs
    custom_category_ID = marketing_items[marketing_df['Category ID'].notna()]

    # Add in reference to Marketing Push for custom IDs, if no reference, then just refer to previous indicated values
    final_sorted = pd.merge(final_sorted, custom_category_ID, on="Article", how="left")
    final_sorted["Final Category ID"] = final_sorted["Category ID_y"].fillna(final_sorted["Category ID_x"])

    # Drop unnecessary columns, rename Final Category ID and rearrange ID for Final DF
    final_sorted.drop(["Year", "Week", "Category", "Category ID_x", "Category ID_y"], axis=1, inplace=True)
    final_sorted.rename(columns={"Final Category ID": "Category ID"}, inplace=True)
    final_sorted = final_sorted[["Product ID", "Category ID", "Catalog ID", "Values", "Article", "Photo"]]

    return final_sorted
    # # Load the existing Excel file
    # existing_workbook = load_workbook(all_categories_file_path)
    # existing_worksheet = existing_workbook.active
    # worksheet_df = pd.DataFrame(existing_worksheet.values)
    # worksheet_df.columns = worksheet_df.iloc[0]     # Dataframe column names are taken from first row
    # worksheet_df = worksheet_df[1:]                 # Data from the other rows of the dataframe 

    # # Loop through the rows of the DataFrame and split the data into sheets based on the 'Category ID' values
    # sheets = {}
    # for _, row in worksheet_df.iterrows():
    #     category_id = row['Category ID']
    #     if category_id not in sheets:
    #         sheets[category_id] = pd.DataFrame(columns=worksheet_df.columns)
    #     sheets[category_id] = pd.concat([sheets[category_id], row.to_frame().T])

    # # Create a new workbook object for the output file
    # output_workbook = Workbook()

    # # Loop through the sheets and add them to the output workbook
    # for category_id, sheet_df in sheets.items():
    #     sheet_df.drop_duplicates(subset=['Article'], keep='first', inplace=True)
    #     output_worksheet = output_workbook.create_sheet(category_id)
    #     row_index = 0

    #     # Loop through each row in the sheet DataFrame
    #     for index, row in sheet_df.iterrows():
    #         article_number = row['Article']
    #         directory = 'https://images.e-charleskeith.com/Article/'
    #         image_path = directory + str(article_number) + '.jpg'

    #         try: 
    #             response = requests.get(image_path, verify=False, timeout = (30, 30))

    #             if response.status_code == 200: 
    #                 pil_image = PILImage.open(BytesIO(response.content))
    #                 image_width, image_height = pil_image.size

    #                 if image_height > image_width:
    #                     pil_image = pil_image.rotate(90)

    #                 # Convert PIL image back to bytes
    #                 image_bytes = BytesIO()
    #                 rgb_img = pil_image.convert('RGB')
    #                 rgb_img.save(image_bytes, format='JPEG')
    #                 image_bytes.seek(0)
                    
    #                 img = Image(image_bytes)
                    
    #                 column_width = 2.0 / 2.54 * 160
    #                 row_height = 3.0 / 2.54 * 64

    #                 img.width = int(column_width)
    #                 img.height = int(row_height)

    #                 cell = 'F{}'.format(row_index + 2)
    #                 output_worksheet.add_image(img, cell)

    #         except requests.Timeout:
    #             print(f"The request timed out for article {article_number}")
    #             continue
    #         except requests.RequestException as e:
    #             print(f"An error occurred for article {article_number}: {e}")
    #             continue
            
    #         row_index += 1

    #     for i in range(1, sheet_df.shape[0] + 2):
    #         output_worksheet.row_dimensions[i].height = 59.5
    #         for col in ['E', 'F']:
    #             output_worksheet.column_dimensions[col].width = 18

    #     data = sheet_df.values.tolist()
    #     output_worksheet.append(sheet_df.columns.tolist())
    #     for row in data:
    #         output_worksheet.append(row)

    # output_workbook.remove(output_workbook['Sheet'])     # Remove 'sheet' from excel file

    # return dcc.send_data_frame(df.to_excel, "mydf.xlsx", sheet_name="Sheet_name_1")