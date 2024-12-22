from modules.utils import *

def parse_exam_data(file_path: str, num_answers: int, delimiter: str = " ") -> pd.DataFrame:
    """
    Parses exam data from a file and converts it into a DataFrame for Excel generation.

    :param file_path: Path to the input file containing the data.
    :param num_answers: Number of answers per ID (constant across all IDs).
    :param delimiter: Delimiter used in the file to separate ID and answers (default is space).
    :return: A pandas DataFrame with parsed data.
    """
    parsed_data = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(delimiter)
                l = len(parts)
                if len(parts) != num_answers + 1:
                    raise ValueError(f"Invalid line format: {line.strip()}")
            
                id_ = parts[0]
                answers = parts[1:]
                
                parsed_data.append({"ID": id_, **{f"Q{i+1}": ans for i, ans in enumerate(answers)}})
    
    except FileNotFoundError:
        print(f"Error: File not found at path {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error while parsing file: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(parsed_data)
    
    df.replace("-", "", inplace=True)  # empty cell
    df.replace("?", "R", inplace=True)  # empty cell (red background)
    
    return df
