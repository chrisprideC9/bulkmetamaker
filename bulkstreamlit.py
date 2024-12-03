import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# Function to assign priority based on URL path
def assign_priority(url):
    if isinstance(url, str):
        if '/collection/' in url:
            return 1
        elif '/products/' in url:
            return 2
        elif '/pages/' in url:
            return 3
        elif '/blogs/' in url:
            return 4
    return 5  # Default priority

# Function to process Meta Titles
def process_meta_titles(df_titles):
    # Remove Non-Indexable rows
    df_titles = df_titles[df_titles['Indexability'].str.lower() != 'non-indexable']

    # Convert Pixel Width to numeric, coercing errors to NaN
    df_titles['Title 1 Pixel Width'] = pd.to_numeric(df_titles['Title 1 Pixel Width'], errors='coerce')

    # Drop rows with Pixel Width < 601
    df_titles = df_titles[df_titles['Title 1 Pixel Width'] >= 601]

    # Identify duplicates
    df_titles['Duplicate_Title'] = df_titles.duplicated(subset=['Title 1'], keep=False)

    # Identify missing titles
    df_titles['Missing_Title'] = df_titles['Title 1'].isnull() | df_titles['Title 1'].str.strip().eq('')

    # Assign priority based on URL
    df_titles['Priority_Title'] = df_titles['Address'].apply(assign_priority)

    return df_titles

# Function to process Meta Descriptions
def process_meta_descriptions(df_desc):
    # Remove Non-Indexable rows
    df_desc = df_desc[df_desc['Indexability'].str.lower() != 'non-indexable']

    # Convert Pixel Width to numeric, coercing errors to NaN
    df_desc['Meta Description 1 Pixel Width'] = pd.to_numeric(df_desc['Meta Description 1 Pixel Width'], errors='coerce')

    # Drop rows with Pixel Width < 961
    df_desc = df_desc[df_desc['Meta Description 1 Pixel Width'] >= 961]

    # Identify duplicates
    df_desc['Duplicate_Description'] = df_desc.duplicated(subset=['Meta Description 1'], keep=False)

    # Identify missing descriptions
    df_desc['Missing_Description'] = df_desc['Meta Description 1'].isnull() | df_desc['Meta Description 1'].str.strip().eq('')

    # Assign priority based on URL
    df_desc['Priority_Description'] = df_desc['Address'].apply(assign_priority)

    return df_desc

# Function to merge and prioritize
def merge_and_prioritize(df_titles, df_desc):
    # Merge on Address
    df = pd.merge(df_titles, df_desc, on='Address', how='outer', suffixes=('_title', '_desc'))

    # Handle URLs present only in titles or descriptions
    df['Priority_Title'] = df['Priority_Title'].fillna(df['Priority_Description'])
    df['Priority_Description'] = df['Priority_Description'].fillna(df['Priority_Title'])

    # Assign weights based on missing and duplicate meta data
    df['Weight'] = 0.0
    df['Missing_Title'] = df['Missing_Title'].fillna(False)
    df['Missing_Description'] = df['Missing_Description'].fillna(False)
    df['Duplicate_Title'] = df['Duplicate_Title'].fillna(False)
    df['Duplicate_Description'] = df['Duplicate_Description'].fillna(False)

    df.loc[df['Missing_Title'], 'Weight'] += 0.50
    df.loc[df['Missing_Description'], 'Weight'] += 0.25
    df.loc[df['Duplicate_Title'], 'Weight'] += 0.50
    df.loc[df['Duplicate_Description'], 'Weight'] += 0.25

    # Assign priority based on URL paths
    df['Overall_Priority'] = df[['Priority_Title', 'Priority_Description']].min(axis=1)

    # Sort by Overall Priority and Weight
    df = df.sort_values(by=['Overall_Priority', 'Weight'], ascending=[True, False])

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df

# Function to assign batch numbers
def assign_batches(df, batch_size):
    df = df.copy()
    df['Batch'] = (df.index // batch_size) + 1
    return df

# Function to create a simple batch list as CSV
def create_simple_batch_csv(df):
    simple_df = df[['Batch', 'Address']].copy()
    simple_df['Batch'] = 'Batch ' + simple_df['Batch'].astype(str)
    return simple_df

# Streamlit App Layout
st.set_page_config(page_title="Bulk Meta Titles & Descriptions Processor", layout="wide")
st.title("Bulk Meta Titles & Descriptions Processor")

st.sidebar.header("Upload CSV Files")

uploaded_titles = st.sidebar.file_uploader("Upload Meta Titles CSV", type=["csv"])
uploaded_desc = st.sidebar.file_uploader("Upload Meta Descriptions CSV", type=["csv"])

if uploaded_titles and uploaded_desc:
    try:
        # Read CSV files
        df_titles = pd.read_csv(uploaded_titles)
        df_desc = pd.read_csv(uploaded_desc)

        # Define required columns
        required_title_cols = ['Address', 'Occurrences', 'Title 1', 'Title 1 Length', 'Title 1 Pixel Width', 'Indexability', 'Indexability Status']
        required_desc_cols = ['Address', 'Occurrences', 'Meta Description 1', 'Meta Description 1 Length', 'Meta Description 1 Pixel Width', 'Indexability', 'Indexability Status']

        # Check for required columns in titles
        if not all(col in df_titles.columns for col in required_title_cols):
            st.error("Meta Titles CSV is missing required columns.")
        # Check for required columns in descriptions
        elif not all(col in df_desc.columns for col in required_desc_cols):
            st.error("Meta Descriptions CSV is missing required columns.")
        else:
            st.success("Files uploaded successfully!")

            # Data Cleaning: Drop rows with NA in critical columns to prevent masking errors
            # Define critical columns for titles and descriptions
            critical_title_cols = ['Indexability', 'Title 1 Pixel Width', 'Title 1']
            critical_desc_cols = ['Indexability', 'Meta Description 1 Pixel Width', 'Meta Description 1']

            initial_titles_count = len(df_titles)
            initial_desc_count = len(df_desc)

            df_titles.dropna(subset=critical_title_cols, inplace=True)
            df_desc.dropna(subset=critical_desc_cols, inplace=True)

            num_dropped_titles = initial_titles_count - len(df_titles)
            num_dropped_desc = initial_desc_count - len(df_desc)

            if num_dropped_titles > 0:
                st.warning(f"Dropped {num_dropped_titles} rows from Meta Titles due to missing critical data.")
            if num_dropped_desc > 0:
                st.warning(f"Dropped {num_dropped_desc} rows from Meta Descriptions due to missing critical data.")

            # Process the data
            df_titles_processed = process_meta_titles(df_titles)
            df_desc_processed = process_meta_descriptions(df_desc)

            # Merge and prioritize
            final_df = merge_and_prioritize(df_titles_processed, df_desc_processed)

            st.subheader("Processed Data")
            st.dataframe(final_df)

            # Batching Options
            st.sidebar.header("Batching Options")
            batch_size = st.sidebar.number_input("Enter batch size (number of items per batch)", min_value=1, value=100, step=1)

            if st.sidebar.button("Create Batches"):
                if batch_size <= 0:
                    st.error("Batch size must be a positive integer.")
                else:
                    final_df_with_batches = assign_batches(final_df, batch_size)
                    num_batches = final_df_with_batches['Batch'].max()

                    st.success(f"Created {int(num_batches)} batch{'es' if num_batches >1 else ''} with up to {batch_size} items each.")

                    # Display data grouped by batches
                    st.subheader("Batched Data Preview")
                    for batch_num in range(1, int(num_batches) + 1):
                        with st.expander(f"Batch {int(batch_num)}"):
                            batch_df = final_df_with_batches[final_df_with_batches['Batch'] == batch_num]
                            st.dataframe(batch_df)

                    # Export Options
                    st.subheader("Export Options")

                    # Option 1: Download All Batches as CSV with Batch Numbers
                    csv_batched = final_df_with_batches.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download All Batches as CSV",
                        data=csv_batched,
                        file_name="batched_meta_data.csv",
                        mime='text/csv',
                    )

                    # Option 2: Download Simple Batch List as CSV
                    simple_batched_csv = create_simple_batch_csv(final_df_with_batches)
                    csv_simple = simple_batched_csv.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Simple Batch List as CSV",
                        data=csv_simple,
                        file_name="simple_batched_list.csv",
                        mime='text/csv',
                    )

    except pd.errors.EmptyDataError:
        st.error("One of the uploaded CSV files is empty. Please upload valid CSV files.")
    except pd.errors.ParserError:
        st.error("Error parsing the CSV files. Please ensure they are properly formatted.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Please upload both Meta Titles and Meta Descriptions CSV files.")

# Footer
st.markdown("---")
st.markdown("© 2024 Calibre Nine | [GitHub Repository](https://github.com/chrisprideC9)")
