from modules.utils import *

def generate_excel_from_csv(csv_path: str, output_excel_path: str):
    """
    Converts a CSV file into an Excel file with specific formatting.
    
    :param csv_path: Path to the input CSV file.
    :param output_excel_path: Path to the output Excel file.
    """
    try:
        df = pd.read_csv(csv_path)
        
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Exam Data"
        
        red_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
        
        # headers
        for col_index, column_name in enumerate(df.columns, start=1):
            sheet.cell(row=1, column=col_index, value=column_name)
        
        # data
        for row_index, row in df.iterrows():
            for col_index, value in enumerate(row, start=1):
                cell = sheet.cell(row=row_index + 2, column=col_index)  # +2 to account for header row
                
                if pd.isna(value):  # blank
                    cell.value = None
                elif value == 'R':  # red background
                    cell.value = None
                    cell.fill = red_fill
                else:  # integer
                    cell.value = (value)
        
        workbook.save(output_excel_path)
        print(f"Excel file generated successfully at: {output_excel_path}")
    
    except FileNotFoundError:
        print(f"Error: File not found at path {csv_path}")
    except Exception as e:
        print(f"Error while generating Excel file: {e}")