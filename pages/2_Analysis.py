import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Navigation Logic ---
# Check if page navigation is needed each time the script runs
if st.session_state.get("navigate_to_main"):
    # Reset flag to prevent repetition
    st.session_state.navigate_to_main = False
    # Go to main page
    st.switch_page("Talep_Tahmin.py")


st.set_page_config(layout="wide", page_title="Prediction Results Analysis")

st.title("Prediction Results Analysis and Visualization")

st.write("""
On this page, you can interactively examine the prediction results generated on the main page by store and product.
""")

# Page navigation function (now only sets a flag)
def go_to_main_page():
    st.session_state.navigate_to_main = True

# Check and get prediction data from session state
if 'final_prediction_df' not in st.session_state:
    st.warning("Please first go to the main page and generate predictions using the 'Generate Predictions' button.")
    st.info("After predictions are generated, results will appear on this page.")
    
    st.button("Return to Demand Prediction Page", on_click=go_to_main_page, type="primary")

else:
    # Get data from session state
    df = st.session_state['final_prediction_df'].copy()
    
    # Convert date index to column
    df.reset_index(inplace=True)

    # 1. Interactive Chart
    st.header("Interactive Prediction Chart by Store and Product")
    st.write("Use the selection boxes below to filter prediction results for a specific store and product.")

    col1, col2 = st.columns(2)
    with col1:
        store_list = sorted(df['store_id'].unique())
        selected_store = st.selectbox("Select Store", store_list)
    with col2:
        item_list = sorted(df[df['store_id'] == selected_store]['item_id'].unique())
        selected_item = st.selectbox("Select Product", item_list)

    filtered_df = df[(df['store_id'] == selected_store) & (df['item_id'] == selected_item)]

    if not filtered_df.empty:
        fig = px.line(
            filtered_df, x='date', y='sales_predicted',
            title=f"Sales Predictions for Store {selected_store} / Product {selected_item}",
            labels={'date': 'Date', 'sales_predicted': 'Predicted Sales Amount'},
            markers=True
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')), line=dict(width=3))
        fig.update_layout(xaxis_title="Date", yaxis_title="Predicted Sales")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data found to display for the selected filters.")

    st.divider()

    # 2. Overview Facet Chart
    st.header("Overview: All Stores and Products")
    st.write("This chart shows predictions for all products in all stores together.")
    
    df_plot = df.copy()
    df_plot['item_id'] = df_plot['item_id'].astype(str)
    df_plot['store_id'] = df_plot['store_id'].astype(str)

    facet_fig = px.line(
        df_plot, x='date', y='sales_predicted', color='item_id',
        facet_col='store_id', facet_col_wrap=5,
        title='Sales Predictions for All Stores and Products',
        labels={'date': 'Date', 'sales_predicted': 'Predicted Sales', 'item_id': 'Product ID'},
        height=800
    )
    facet_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(facet_fig, use_container_width=True)
    
    st.divider()

    # 3. Heatmap
    st.header("Heatmap: Product Sales Intensity")
    st.write("Use the heatmap below to see the sales intensity of all products in a specific store.")

    heatmap_store_list = sorted(df['store_id'].unique())
    heatmap_selected_store = st.selectbox("Select Store for Heatmap", heatmap_store_list, key="heatmap_store_select")

    heatmap_df = df[df['store_id'] == heatmap_selected_store]

    if not heatmap_df.empty:
        try:
            pivot_df = heatmap_df.pivot_table(index='item_id', columns='date', values='sales_predicted')
            heatmap_fig = px.imshow(
                pivot_df,
                labels=dict(x="Date", y="Product ID", color="Predicted Sales"),
                x=pivot_df.columns.strftime('%Y-%m-%d'), y=pivot_df.index,
                aspect="auto",
                title=f"Product Sales Prediction Heatmap for Store {heatmap_selected_store}"
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred while creating the heatmap: {e}")
    else:
        st.info("No data found to create heatmap for this store.")

    st.divider()

    # --- 4. Combined Sales and Prediction Chart ---
    st.header("Combined Historical Sales and Predictions Chart")
    st.write("View historical actual sales and model predictions for a specific store and product in a single chart.")

    # Get historical data too
    historical_df_from_session = st.session_state.get('historical_df')
    if historical_df_from_session is not None:
        # Solution to the error: Take a copy of the data before processing to preserve the original.
        historical_df = historical_df_from_session.copy()
        historical_df.reset_index(inplace=True)

        col3, col4 = st.columns(2)
        with col3:
            store_list_combo = sorted(df['store_id'].unique())
            selected_store_combo = st.selectbox("Select Store", store_list_combo, key="combo_store_select")
        with col4:
            item_list_combo = sorted(df[df['store_id'] == selected_store_combo]['item_id'].unique())
            selected_item_combo = st.selectbox("Select Product", item_list_combo, key="combo_item_select")
        
        # Filter historical data for selected store/product
        history_filtered = historical_df[
            (historical_df['store_id'] == selected_store_combo) &
            (historical_df['item_id'] == selected_item_combo)
        ].copy()
        history_filtered = history_filtered[['date', 'sales']]
        history_filtered.rename(columns={'sales': 'value'}, inplace=True)
        history_filtered['type'] = 'Actual Sales'

        # Filter prediction data for selected store/product
        prediction_filtered = df[
            (df['store_id'] == selected_store_combo) &
            (df['item_id'] == selected_item_combo)
        ].copy()
        prediction_filtered = prediction_filtered[['date', 'sales_predicted']]
        prediction_filtered.rename(columns={'sales_predicted': 'value'}, inplace=True)
        prediction_filtered['type'] = 'Predicted Sales'
        
        # Combine the two dataframes
        combined_plot_df = pd.concat([history_filtered, prediction_filtered], ignore_index=True)
        
        if not combined_plot_df.empty:
            combo_fig = px.line(
                combined_plot_df,
                x='date',
                y='value',
                color='type',
                title=f"Store {selected_store_combo} / Product {selected_item_combo}: Sales and Prediction Trend",
                labels={'date': 'Date', 'value': 'Sales Amount', 'type': 'Data Type'}
            )
            # Show prediction start point with a vertical line
            prediction_start_date = prediction_filtered['date'].min()
            
            # TypeError solution: Remove the problematic text label.
            # Red dashed line is sufficient to show prediction start.
            combo_fig.add_vline(x=prediction_start_date.to_pydatetime(), line_width=2, line_dash="dash", line_color="red")
            st.plotly_chart(combo_fig, use_container_width=True)
        else:
            st.warning("No data found to create combined chart for this selection.")

    st.divider()
    st.button("Return to Demand Prediction Page", on_click=go_to_main_page)
