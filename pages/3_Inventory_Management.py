import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(layout="wide", page_title="Inventory Management Analysis")

st.title("Sales Growth Analysis for Inventory Management")

st.write("""
This page calculates and visualizes the expected sales growth for a selected product and week in the first quarter of 2018 compared to the same week last year, by store.
This analysis can help with your inventory planning.
""")

# Check if required data is in session state
if 'final_prediction_df' not in st.session_state or 'historical_df' not in st.session_state:
    st.warning("To view this page, please first create a prediction from the 'Demand Prediction' page.")
    
    # Page navigation button
    if st.button("Go to Demand Prediction Page"):
        st.switch_page("Talep_Tahmin.py")
else:
    # Get data from session state by copying
    predictions_df = st.session_state['final_prediction_df'].copy()
    historical_df = st.session_state['historical_df'].copy()

    # Convert date indexes to columns
    if 'date' not in predictions_df.columns:
        predictions_df.reset_index(inplace=True)
    if 'date' not in historical_df.columns:
        historical_df.reset_index(inplace=True)

    st.header("1. Select Analysis Criteria")

    # --- Criteria Selection Area ---
    # Product selection
    all_items = sorted(historical_df['item_id'].unique())
    selected_item = st.selectbox("Select Product for Analysis:", all_items)

    # Period Type Selection
    period_type = st.radio(
        "Select Analysis Period:",
        ('Weekly', 'Monthly', 'Quarterly'),
        horizontal=True,
        key="period_select"
    )

    # Dynamic Period Selection UI
    start_date = None
    end_date = None
    period_label = ""

    if period_type == 'Weekly':
        selected_date = st.date_input(
            "Select Start of Analysis Week:",
            value=date(2018, 1, 1),
            min_value=date(2018, 1, 1),
            max_value=date(2018, 3, 31) - timedelta(days=6),
            key="weekly_date_select"
        )
        start_date = selected_date
        end_date = selected_date + timedelta(days=6)
        period_label = f"{start_date.strftime('%d %b')} - {end_date.strftime('%d %b %Y')}"

    elif period_type == 'Monthly':
        month_map = {'January': 1, 'February': 2, 'March': 3}
        selected_month_name = st.selectbox("Select Month:", list(month_map.keys()), key="monthly_select")
        selected_month = month_map[selected_month_name]
        start_date = date(2018, selected_month, 1)
        # Find last day of month
        next_month_start = date(2018, selected_month + 1, 1) if selected_month < 3 else date(2018, 4, 1)
        end_date = next_month_start - timedelta(days=1)
        period_label = f"{selected_month_name} {start_date.year}"

    elif period_type == 'Quarterly':
        start_date = date(2018, 1, 1)
        end_date = date(2018, 3, 31)
        period_label = "Q1 2018"
    
    st.info(f"Selected analysis range: **{period_label}**")

    # Calculation button
    if st.button("Calculate Sales Growth", type="primary"):
        # Ensure start_date and end_date are not None
        if start_date is None or end_date is None:
            st.error("Please select a valid period and date range.")
            st.stop()

        st.header(f"2. Store-Based Sales Growth Results")
        
        # Calculate same period last year (more robust method)
        hist_start_date = pd.to_datetime(start_date) - pd.DateOffset(years=1)
        hist_end_date = pd.to_datetime(end_date) - pd.DateOffset(years=1)

        # 1. Calculate predicted sales
        mask_predict = (predictions_df['item_id'] == selected_item) & (predictions_df['date'] >= pd.to_datetime(start_date)) & (predictions_df['date'] <= pd.to_datetime(end_date))
        predicted_sales = predictions_df[mask_predict].groupby('store_id')['sales_predicted'].sum().reset_index()
        predicted_sales.rename(columns={'sales_predicted': f'Predicted Sales ({period_label})'}, inplace=True)
        
        # 2. Calculate historical sales
        mask_hist = (historical_df['item_id'] == selected_item) & (historical_df['date'] >= hist_start_date) & (historical_df['date'] <= hist_end_date)
        historical_sales = historical_df[mask_hist].groupby('store_id')['sales'].sum().reset_index()
        hist_period_label = f"{hist_start_date.strftime('%b %Y')}" if period_type != 'Weekly' else f"{hist_start_date.strftime('%d %b')} - {hist_end_date.strftime('%d %b %Y')}"
        historical_sales.rename(columns={'sales': f'Last Year Sales ({hist_period_label})'}, inplace=True)

        if predicted_sales.empty:
            st.warning("No prediction data found for the selected period. Please select a different period.")
        else:
            # 3. Merge two datasets and calculate growth
            comparison_df = pd.merge(predicted_sales, historical_sales, on='store_id', how='left').fillna(0)
            comparison_df['Growth Amount'] = comparison_df[f'Predicted Sales ({period_label})'] - comparison_df[f'Last Year Sales ({hist_period_label})']
            
            st.subheader(f"Expected Sales Change by Store for Product {selected_item} ({period_type})")
            
            fig = px.bar(
                comparison_df.sort_values(by="Growth Amount", ascending=False),
                x='store_id',
                y='Growth Amount',
                color='Growth Amount',
                color_continuous_scale=px.colors.sequential.RdBu_r,
                labels={'store_id': 'Store ID', 'Growth Amount': f'Expected Sales Difference (Units)'},
                title=f"{period_type} Sales Difference Prediction for Product {selected_item}"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Detailed Data")
            st.dataframe(comparison_df)

    st.divider()

    st.header("2. Products Expected to Have Demand Growth (2018 Q1)")
    st.write("This analysis identifies products expected to show the highest demand growth across all stores by comparing total predicted sales in Q1 2018 with the same period in 2017.")

    # Calculate total predictions for Q1 2018 by product
    q1_2018_start, q1_2018_end = date(2018, 1, 1), date(2018, 3, 31)
    mask_predict_q1 = (pd.to_datetime(predictions_df['date']).dt.date >= q1_2018_start) & (pd.to_datetime(predictions_df['date']).dt.date <= q1_2018_end)
    predicted_sales_q1 = predictions_df[mask_predict_q1].groupby('item_id')['sales_predicted'].sum().reset_index()
    predicted_sales_q1.rename(columns={'sales_predicted': 'Total Prediction (2018 Q1)'}, inplace=True)

    # Calculate total historical sales for Q1 2017 by product
    q1_2017_start, q1_2017_end = date(2017, 1, 1), date(2017, 3, 31)
    mask_hist_q1 = (pd.to_datetime(historical_df['date']).dt.date >= q1_2017_start) & (pd.to_datetime(historical_df['date']).dt.date <= q1_2017_end)
    historical_sales_q1 = historical_df[mask_hist_q1].groupby('item_id')['sales'].sum().reset_index()
    historical_sales_q1.rename(columns={'sales': 'Total Sales (2017 Q1)'}, inplace=True)

    # Merge data and calculate growth
    product_growth_df = pd.merge(predicted_sales_q1, historical_sales_q1, on='item_id', how='left').fillna(0)
    product_growth_df['Growth Amount'] = product_growth_df['Total Prediction (2018 Q1)'] - product_growth_df['Total Sales (2017 Q1)']
    
    # --- Calculate Average of All Products ---
    total_avg_growth = product_growth_df['Growth Amount'].mean()

    # Select top 15 products with highest growth
    top_growing_products = product_growth_df.sort_values(by="Growth Amount", ascending=False).head(15)

    st.subheader("Top 15 Products Expected to Grow Most (All Stores)")
    
    fig_growth = px.bar(
        top_growing_products,
        x='item_id',
        y='Growth Amount',
        color='Growth Amount',
        color_continuous_scale='Greens',
        labels={'item_id': 'Product ID', 'Growth Amount': 'Expected Total Sales Growth (Units)'},
        title="Products Expected to Have Highest Demand Growth in Q1 2018 (All Stores)"
    )
    # Add average line
    fig_growth.add_hline(
        y=total_avg_growth, line_dash="dot",
        annotation_text=f"All Products Average: {total_avg_growth:,.0f}",
        annotation_position="bottom right"
    )
    fig_growth.update_xaxes(type='category')
    st.plotly_chart(fig_growth, use_container_width=True)

    # --- Products Expected to Decline Most ---
    st.subheader("Top 15 Products Expected to Decline Most (All Stores)")

    # Select top 15 products with highest decline
    top_declining_products = product_growth_df.nsmallest(15, 'Growth Amount')

    # Reverse color scale to better show negative values
    fig_decline = px.bar(
        top_declining_products,
        x='item_id',
        y='Growth Amount',
        color='Growth Amount',
        color_continuous_scale=px.colors.sequential.Reds_r, # Red tones
        labels={'item_id': 'Product ID', 'Growth Amount': 'Expected Total Sales Difference (Units)'},
        title="Products Expected to Have Highest Demand Decline in Q1 2018 (All Stores)"
    )
    # Add average line
    fig_decline.add_hline(
        y=total_avg_growth, line_dash="dot",
        annotation_text=f"All Products Average: {total_avg_growth:,.0f}",
        annotation_position="top right"
    )
    fig_decline.update_xaxes(type='category')
    st.plotly_chart(fig_decline, use_container_width=True)


    st.divider()

    st.header("3. Products Expected to Have Demand Growth by Store")
    st.write("Select a store to analyze the products expected to have the highest demand growth.")

    # Store filter
    all_stores = sorted(historical_df['store_id'].unique())
    selected_store_growth = st.selectbox("Select Store:", all_stores, key="store_growth_select")

    if selected_store_growth:
        # Filter data for selected store
        predictions_df_store = predictions_df[predictions_df['store_id'] == selected_store_growth]
        historical_df_store = historical_df[historical_df['store_id'] == selected_store_growth]

        # Store-specific Q1 2018 predictions
        mask_predict_q1_store = (pd.to_datetime(predictions_df_store['date']).dt.date >= q1_2018_start) & (pd.to_datetime(predictions_df_store['date']).dt.date <= q1_2018_end)
        predicted_sales_q1_store = predictions_df_store[mask_predict_q1_store].groupby('item_id')['sales_predicted'].sum().reset_index()
        predicted_sales_q1_store.rename(columns={'sales_predicted': 'Total Prediction (2018 Q1)'}, inplace=True)

        # Store-specific Q1 2017 historical sales
        mask_hist_q1_store = (pd.to_datetime(historical_df_store['date']).dt.date >= q1_2017_start) & (pd.to_datetime(historical_df_store['date']).dt.date <= q1_2017_end)
        historical_sales_q1_store = historical_df_store[mask_hist_q1_store].groupby('item_id')['sales'].sum().reset_index()
        historical_sales_q1_store.rename(columns={'sales': 'Total Sales (2017 Q1)'}, inplace=True)

        # Merge data and calculate growth
        product_growth_df_store = pd.merge(predicted_sales_q1_store, historical_sales_q1_store, on='item_id', how='left').fillna(0)
        product_growth_df_store['Growth Amount'] = product_growth_df_store['Total Prediction (2018 Q1)'] - product_growth_df_store['Total Sales (2017 Q1)']
        
        top_growing_products_store = product_growth_df_store.sort_values(by="Growth Amount", ascending=False).head(15)

        if not top_growing_products_store.empty:
            st.subheader(f"Top 15 Products Expected to Grow Most in Store {selected_store_growth}")
            fig_growth_store = px.bar(
                top_growing_products_store,
                x='item_id',
                y='Growth Amount',
                color='Growth Amount',
                color_continuous_scale='Greens',
                labels={'item_id': 'Product ID', 'Growth Amount': 'Expected Total Sales Growth (Units)'},
                title=f"Products Expected to Have Highest Demand Growth in Q1 2018 for Store {selected_store_growth}"
            )
            fig_growth_store.update_xaxes(type='category')
            st.plotly_chart(fig_growth_store, use_container_width=True)
        else:
            st.info(f"No growth data found for store {selected_store_growth}.")
        
        st.subheader("Growth Data for All Products")
        st.dataframe(product_growth_df.sort_values(by="Growth Amount", ascending=False))


    st.divider()
    st.header("4. Stores Expected to Have Highest Demand Growth (2018 Q1)")

    # --- Store-Based Quarterly Analysis ---
    # Q1 2018 predictions by store
    q1_predicted_sales_store = predictions_df[
        (pd.to_datetime(predictions_df['date']).dt.date >= q1_2018_start) & (pd.to_datetime(predictions_df['date']).dt.date <= q1_2018_end)
    ].groupby('store_id')['sales_predicted'].sum().reset_index()
    q1_predicted_sales_store.rename(columns={'sales_predicted': 'Q1 2018 Prediction'}, inplace=True)

    # Q1 2017 historical sales by store
    q1_historical_sales_store = historical_df[
        (pd.to_datetime(historical_df['date']).dt.date >= q1_2017_start) & (pd.to_datetime(historical_df['date']).dt.date <= q1_2017_end)
    ].groupby('store_id')['sales'].sum().reset_index()
    q1_historical_sales_store.rename(columns={'sales': 'Q1 2017 Sales'}, inplace=True)

    # Merge data and calculate growth
    store_growth_df = pd.merge(q1_predicted_sales_store, q1_historical_sales_store, on='store_id', how='left').fillna(0)
    store_growth_df['Growth Amount'] = store_growth_df['Q1 2018 Prediction'] - store_growth_df['Q1 2017 Sales']

    # Calculate average growth for all stores
    total_avg_store_growth = store_growth_df['Growth Amount'].mean()

    # --- Stores Expected to Grow Most ---
    st.subheader("Top 15 Stores Expected to Have Highest Demand Growth")
    top_growing_stores = store_growth_df.nlargest(15, 'Growth Amount')

    fig_store_growth = px.bar(
        top_growing_stores,
        x='store_id',
        y='Growth Amount',
        color='Growth Amount',
        color_continuous_scale='Greens',
        labels={'store_id': 'Store ID', 'Growth Amount': 'Expected Total Sales Growth (Units)'},
        title="Stores Expected to Have Highest Demand Growth in Q1 2018"
    )
    fig_store_growth.add_hline(
        y=total_avg_store_growth, line_dash="dot",
        annotation_text=f"All Stores Average: {total_avg_store_growth:,.0f}",
        annotation_position="bottom right"
    )
    fig_store_growth.update_xaxes(type='category')
    st.plotly_chart(fig_store_growth, use_container_width=True)


    # --- Stores Expected to Decline Most ---
    st.subheader("Top 15 Stores Expected to Have Highest Demand Decline")
    top_declining_stores = store_growth_df.nsmallest(15, 'Growth Amount')

    fig_store_decline = px.bar(
        top_declining_stores,
        x='store_id',
        y='Growth Amount',
        color='Growth Amount',
        color_continuous_scale=px.colors.sequential.Reds_r,
        labels={'store_id': 'Store ID', 'Growth Amount': 'Expected Total Sales Decline (Units)'},
        title="Stores Expected to Have Highest Demand Decline in Q1 2018"
    )
    fig_store_decline.add_hline(
        y=total_avg_store_growth, line_dash="dot",
        annotation_text=f"All Stores Average: {total_avg_store_growth:,.0f}",
        annotation_position="top right"
    )
    fig_store_decline.update_xaxes(type='category')
    st.plotly_chart(fig_store_decline, use_container_width=True) 