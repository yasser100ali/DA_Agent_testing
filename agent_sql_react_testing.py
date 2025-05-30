
from agents import Agent, ReAct
import json
import streamlit as st
import sqlite3
import pandas as pd
import time


DATABASE_FILE = "testing_data/chinook.db"
#DATABASE_FILE = "testing_data/nba.sqlite"

class sqlAgentReAct:
    def __init__(self, user_input):
        self.user_input = user_input 


    # still need to integrate this, thought jury is still out whether I actually need to 
    def orchestrator(self):
        system_prompt = """You are an orchestrator agent. Your job is to figure out if the given user prompt could be solved in a single step using one 
        of the listed functions below or if requires a more dynamic approach with custom python sqlite code (coder agent). The coder agent is ReAct based so it could iteratively solve a problem.

        Your two options are 1. 'function_call' and 2. 'coder_agent'

        output your answer as a json.

        example 
        ```json
        {
            'agent': 'function_call' or 'coder_agent' # depending on the task, CHOOSE ONE
        }
        ```
        """

        json_output = Agent(self.user_input, {}).json_agent(system_prompt, "", self.user_input)
        return json_output

    def sql_agent(self):

        reason_job = """
        Your job is to direct the act portion of the ReAct (you are Reason) to solve the user's request.
        Your act agent will use sqlite3 within python, so direct your plan and directions according to that. 

        
        """

        act_job = f"""
        You are a python agent, specifically specializing in sqlite3. 

        You are part of a ReAct loop and will follow directions according to Reason's dictates.
        when connecting to the database, the path is {DATABASE_FILE}

        Return your output as a JSON where you write the important details of various aspects of what you found. 

        Here's what an output code could look like: 
        """
        act_job += """
        ```python 
        import sqlite3
        import pandas as pd
        import os # To check if the database file exists

        def main():

            # Connects to the Chinook database, executes SQLite examples (commented),
            # runs a query to fetch Rock tracks, and returns the result as a Pandas DataFrame.
            db_path = 'chinook.db'

            if not os.path.exists(db_path):
                print(f"Error: Database file '{db_path}' not found.")
                print("Please download the Chinook database and place it in the same directory as this script.")
                print("You can find it here: https://www.sqlitetutorial.net/sqlite-sample-database/")
                return

            conn = None  # Initialize conn to None
            try:
                # Connect to the SQLite database
                conn = sqlite3.connect(db_path)
                print(f"Successfully connected to {db_path}")

                # --- Begin SQLite Code Examples (as comments) ---

                # 1. Select all artists:
                # CURSOR = conn.cursor()
                # CURSOR.execute("SELECT * FROM Artist LIMIT 5;") # Limit for brevity in direct execution
                # print("\nFirst 5 Artists:")
                # for row in CURSOR.fetchall():
                #     print(row)
                # CURSOR.close()

                # 2. Select all albums and their artists:
                
                # SELECT
                #     Album.Title AS AlbumTitle,
                #     Artist.Name AS ArtistName
                # FROM
                #     Album
                # INNER JOIN
                #     Artist ON Album.ArtistId = Artist.ArtistId;
                

                # 3. Find all tracks by the band 'Queen':

                # SELECT
                #     Track.Name AS TrackName,
                #     Album.Title AS AlbumTitle
                # FROM
                #     Track
                # INNER JOIN
                #     Album ON Track.AlbumId = Album.AlbumId
                # INNER JOIN
                #     Artist ON Album.ArtistId = Artist.ArtistId
                # WHERE
                #     Artist.Name = 'Queen';
                

                # 4. List customers from the USA:

                # SELECT
                #     FirstName,
                #     LastName,
                #     City,
                #     State,
                #     Country
                # FROM
                #     Customer
                # WHERE
                #     Country = 'USA';


                # 5. Count the number of tracks in each playlist:
        =
                # SELECT
                #     Playlist.Name AS PlaylistName,
                #     COUNT(PlaylistTrack.TrackId) AS NumberOfTracks
                # FROM
                #     Playlist
                # INNER JOIN
                #     PlaylistTrack ON Playlist.PlaylistId = PlaylistTrack.PlaylistId
                # GROUP BY
                #     Playlist.PlaylistId, Playlist.Name
                # ORDER BY
                #     NumberOfTracks DESC;


                # 6. Find the total sales for each country:

                # SELECT
                #     BillingCountry,
                #     SUM(Total) AS TotalSales
                # FROM
                #     Invoice
                # GROUP BY
                #     BillingCountry
                # ORDER BY
                #     TotalSales DESC;

                # # 7. Add a new genre (Example - be cautious with data modification):
                # # Note: For INSERT/UPDATE/DELETE, you'd use conn.execute() and conn.commit()
                
                # -- INSERT INTO Genre (Name) VALUES ('Synthwave');
                

                # # 8. Update a customer's email address (Example):
                
                # -- UPDATE Customer
                # -- SET Email = 'new.email@example.com'
                # -- WHERE CustomerId = 1;
                

                # # 9. Delete a playlist (Example):
                
                # -- DELETE FROM PlaylistTrack
                # -- WHERE PlaylistId = (SELECT PlaylistId FROM Playlist WHERE Name = 'My New Playlist');
                # -- DELETE FROM Playlist
                # -- WHERE Name = 'My New Playlist';
                
                # # --- End SQLite Code Examples ---


                # --- Made-up query to get a table and return it as a DataFrame ---

                # This query selects track names, album titles, and genre names
                # for tracks belonging to the 'Rock' genre.
                # It joins Track, Album, and Genre tables.
                query = '''
                SELECT
                    t.Name AS TrackName,
                    a.Title AS AlbumTitle,
                    g.Name AS GenreName
                FROM
                    Track t
                INNER JOIN
                    Album a ON t.AlbumId = a.AlbumId
                INNER JOIN
                    Genre g ON t.GenreId = g.GenreId
                WHERE
                    g.Name = 'Rock'
                LIMIT 15; -- Limiting results for display purposes
                '''

                # Use pandas to directly execute the query and return a DataFrame
                rock_tracks_df = pd.read_sql_query(query, conn)

                if not rock_tracks_df.empty:
                    print("\\n--- Rock Tracks Found, Converting to JSON ---")
                    # Convert the DataFrame to a JSON string. 
                    # orient='records' creates a list of objects, ideal for JSON.
                    # indent=4 makes the JSON output human-readable.
                    json_output = rock_tracks_df.to_json(orient="records", indent=4)
                    print(json_output)
                else:
                    print("\\nNo tracks found for the 'Rock' genre with the given query.")
                    # Return an empty JSON object as a string if no data is found
                    json_output = json.dumps([]) 

                # Return the result as a JSON string
                return json_output

            except sqlite3.Error as e:
                print(f"SQLite error: {e}")
            except pd.io.sql.DatabaseError as e: # Pandas specific SQL error
                print(f"Pandas SQL query error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            finally:
                if conn:
                    conn.close()
                    print("\nDatabase connection closed.")


        ```
        """

        react_agent = ReAct(self.user_input)
        final_output = react_agent.react_loop(reason_job, act_job, max_iterations=20)
        return final_output
        
    def main(self):
        return self.sql_agent()