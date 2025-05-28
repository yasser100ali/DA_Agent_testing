
import os.path
import base64
import datetime
import pytz # For timezone handling

from email.message import EmailMessage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import traceback

# --- Configuration ---
# If modifying these scopes, delete the file token.json.
# Choose the level of access needed:
# Gmail: .readonly, .send, .modify
# Calendar: .readonly (view), .events (view/edit events), .calendar (full access)
SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',      # Allows reading, sending, modifying Gmail emails
    'https://www.googleapis.com/auth/calendar.events'   # Allows reading and writing Calendar events
]

# File paths (ensure these files are in the same directory as the script)
TOKEN_PATH = 'secretary_token.json' # Stores authorization tokens after first login
CREDENTIALS_PATH = 'credentials.json' # Your downloaded OAuth credentials
# --- End Configuration --

def authenticate_google_apis():
    """Handles OAuth 2.0 authentication flow for specified SCOPES
    using credentials.json and token.json. Builds and returns service
    objects for Gmail and Calendar APIs if successful.
    """
    creds = None
    # Load existing tokens if available
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except Exception as e:
            print(f"Error loading token.json: {e}. Re-authenticating.")
            if os.path.exists(TOKEN_PATH):
                 os.remove(TOKEN_PATH) # Remove corrupt token
            creds = None

    # Check if the current token has all the necessary scopes. If not, force re-auth.
    if creds and not set(SCOPES).issubset(set(creds.scopes)):
        print("Token found, but required scopes are missing. Re-authenticating.")
        if os.path.exists(TOKEN_PATH):
            os.remove(TOKEN_PATH)
        creds = None # Force re-authentication flow

    # If no valid credentials, initiate login/refresh flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # If credentials expired, try to refresh them
            try:
                print("Refreshing expired credentials...")
                creds.refresh(Request())
                print("Credentials refreshed successfully.")
                # Save the refreshed credentials
                try:
                    with open(TOKEN_PATH, 'w') as token:
                        token.write(creds.to_json())
                    print(f"Refreshed credentials saved to {TOKEN_PATH}")
                except Exception as e:
                    print(f"Error saving refreshed token to {TOKEN_PATH}: {e}")
            except Exception as e:
                print(f"Error refreshing token: {e}. Need to re-authenticate.")
                # If refresh fails, delete the token file and force re-authenticate
                if os.path.exists(TOKEN_PATH):
                    os.remove(TOKEN_PATH)
                creds = None # Ensure re-authentication flow runs
        else:
            # Run the full OAuth flow if no token or refresh failed
            try:
                print("No valid credentials found or refresh failed. Starting authentication flow...")
                # Check if credentials file exists
                if not os.path.exists(CREDENTIALS_PATH):
                    print(f"ERROR: credentials.json not found at path: {os.path.abspath(CREDENTIALS_PATH)}")
                    print("Please download your OAuth 2.0 Desktop client credentials from Google Cloud Console and save it as credentials.json in the same directory as this script.")
                    return None, None # Return None for both services

                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_PATH, SCOPES)
                # run_local_server will open a browser tab for user consent
                creds = flow.run_local_server(port=0)
                print("Authentication successful.")
                # Save the newly obtained credentials for future runs
                try:
                    with open(TOKEN_PATH, 'w') as token:
                        token.write(creds.to_json())
                    print(f"Credentials saved to {TOKEN_PATH}")
                except Exception as e:
                    print(f"Error saving token to {TOKEN_PATH}: {e}")

            except FileNotFoundError:
                print(f"ERROR: credentials.json not found at path: {os.path.abspath(CREDENTIALS_PATH)}")
                print("Please download your OAuth 2.0 Desktop client credentials from Google Cloud Console and save it as credentials.json in the same directory as this script.")
                return None, None
            except Exception as e:
                print(f"Error during authentication flow: {e}")
                return None, None

    # Build and return the service objects
    gmail_service = None
    calendar_service = None

    # Build Gmail Service
    try:
        gmail_service = build('gmail', 'v1', credentials=creds)
        print("Gmail API service created successfully.")
    except HttpError as error:
        print(f'An HTTP error occurred building the Gmail service: {error}')
        if error.resp.status == 403:
             print("Gmail Error 403: Check if the Gmail API is enabled in your Google Cloud project.")
             print("Also verify that the Gmail SCOPES requested are appropriate for your OAuth consent screen configuration.")
    except Exception as e:
        print(f'An unexpected error occurred during Gmail service build: {e}')

    # Build Calendar Service
    try:
        calendar_service = build('calendar', 'v3', credentials=creds)
        print("Calendar API service created successfully.")
    except HttpError as error:
        print(f'An HTTP error occurred building the Calendar service: {error}')
        if error.resp.status == 403:
            print("Calendar Error 403: Check if the Google Calendar API is enabled in your Google Cloud project.")
            print("Also verify that the Calendar SCOPES requested are appropriate for your OAuth consent screen configuration.")
    except Exception as e:
         print(f'An unexpected error occurred during Calendar service build: {e}')

    return gmail_service, calendar_service # Return both service objects

# --- Gmail Functions (Unchanged) ---

def read_emails(days=7, max_results=100, query='in:inbox'):
    """Lists emails matching the query and prints basic info."""
    service, _ = authenticate_google_apis()
    if not service:
        print("Gmail service not available for reading emails.")
        return

    try:
        print(f"\nFetching emails with query: '{query}'...")
        results = service.users().messages().list(
            userId='me', 
            q="in:inbox", 
            maxResults=100
        ).execute()
        messages = results.get('messages', [])

        if not messages:
            print(f'No messages found matching query: "{query}".')
            return
        
        messages_list = []

        print(f'Found {len(messages)} messages:')
        for message_info in messages:
            msg_id = message_info['id']
            try:
                msg = service.users().messages().get(userId='me', id=msg_id, format='metadata').execute()
                payload = msg.get('payload', {})
                headers = payload.get('headers', [])

                subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
                date = next((h['value'] for h in headers if h['name'].lower() == 'date'), 'No Date')
                snippet = msg.get('snippet', 'No snippet available.')
                
                message_data = {
                    'id': msg_id, 
                    'date': date,
                    'subject': subject,
                    'sender': sender,
                    'snipper': snippet
                }

                messages_list.append(message_data)

                print(f"\n--- Message ID: {msg_id} ---")
                print(f"  Date: {date}")
                print(f"  From: {sender}")
                print(f"  Subject: {subject}")
                print(f"  Snippet: {snippet}")

        
            except HttpError as error:
                print(f'Error fetching details for message ID {msg_id}: {error}')
            except Exception as e:
                print(f'Unexpected error processing message ID {msg_id}: {e}')

        return str(messages_list)

    except HttpError as error:
        print(f'An error occurred reading emails: {error}')
    except Exception as e:
        print(f'An unexpected error occurred during email read operation: {e}')


def send_email(to, subject, body):
    """Creates and sends an email message."""
    service, _ = authenticate_google_apis()

    if not service:
        print("Gmail service not available for sending email.")
        return None

    try:
        message = EmailMessage()
        message.set_content(body)
        message['To'] = to
        message['Subject'] = subject

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message_request = {'raw': encoded_message}

        # pylint: disable=E1101
        send_message_response = service.users().messages().send(userId='me', body=create_message_request).execute()
        print(f"Message Id: {send_message_response['id']} sent successfully to {to}")
        return send_message_response

    except HttpError as error:
        print(f'An HTTP error occurred sending email: {error}')
        if error.resp.status == 400:
             print("Error 400: Bad Request. Check the format of the 'To' address and the message body.")
        elif error.resp.status == 403:
             print("Error 403: Permission denied. Ensure the SCOPES include sending permissions (e.g., gmail.send or gmail.modify).")
        return None
    except Exception as e:
        print(f'An unexpected error occurred during email send operation: {e}')
        return None

# --- Google Calendar Functions (New) ---

def list_upcoming_events(days=7, max_results=10):
    """Lists the next max_results events on the user's primary calendar."""
    _, service = authenticate_google_apis()

    if not service:
        print("Calendar service not available for listing events.")
        return

    try:

        # 'Z' indicates UTC time
        # Get the current time in UTC as a timezone-aware object
        now_utc_aware = datetime.datetime.now(datetime.timezone.utc)
        time_max_aware = now_utc_aware + datetime.timedelta(days=days)
        # Format it in ISO format including the UTC offset ('Z' for UTC)
        now_utc = now_utc_aware.isoformat()
        time_max_utc = time_max_aware.isoformat()
        print(f'\nGetting the upcoming {max_results} events...')
        # pylint: disable=E1101
        events_result = service.events().list(
            calendarId='primary', # 'primary' is the main calendar for the authenticated user
            timeMin=now_utc,
            timeMax=time_max_utc,
            maxResults=max_results,
            singleEvents=True,    # Treat recurring events as individual instances
            orderBy='startTime'   # Order events chronologically
        ).execute()
        events = events_result.get('items', [])

        if not events:
            print('No upcoming events found.')
            return

        print("Upcoming events:")

        events_list = []
        for event in events:
            start = event['start'] # Handles all-day events
            end = event['end']
            summary = event['summary']


            event_details = {
                'start': start,
                'end': end,
                'event_summary': summary
            }


            events_list.append(event_details)

        return str(events_list)
    
    except HttpError as error:
        print(f'An HTTP error occurred listing calendar events: {error}')
    except Exception as e:
        print(f'An unexpected error occurred during calendar list operation: {e}')
        traceback.print_exc()

def create_calendar_event(summary, start_time_str, end_time_str, timezone='America/Los_Angeles'):
    """Creates an event on the user's primary calendar.

    Args:
        service: Authorized Calendar API service instance.
        summary (str): The title of the event.
        start_time_str (str): Start date/time in RFC3339 format (e.g., '2025-05-05T10:00:00-07:00').
        end_time_str (str): End date/time in RFC3339 format (e.g., '2025-05-05T11:00:00-07:00').
        timezone (str): The IANA timezone name (e.g., 'America/Los_Angeles', 'Europe/London').
    """
    _, service = authenticate_google_apis()

    if not service:
        print("Calendar service not available for creating events.")
        return None

    event = {
        'summary': summary,
        # 'location': 'Optional Location String',
        # 'description': 'Optional Description String',
        'start': {
            'dateTime': start_time_str,
            'timeZone': timezone,
        },
        'end': {
            'dateTime': end_time_str,
            'timeZone': timezone,
        },
        # 'attendees': [
        #     {'email': 'attendee1@example.com'},
        #     {'email': 'attendee2@example.com'},
        # ],
        # 'reminders': { # Optional reminders
        #     'useDefault': False,
        #     'overrides': [
        #         {'method': 'email', 'minutes': 24 * 60}, # Email reminder 1 day before
        #         {'method': 'popup', 'minutes': 10},      # Popup reminder 10 mins before
        #     ],
        # },
    }

    try:
        print(f"\nAttempting to create event: '{summary}' from {start_time_str} to {end_time_str} ({timezone})")
        # pylint: disable=E1101
        created_event = service.events().insert(
            calendarId='primary', # Use 'primary' calendar
            body=event
        ).execute()
        print(f"Event created successfully: {created_event.get('htmlLink')}")
        return created_event

    except HttpError as error:
        print(f'An HTTP error occurred creating calendar event: {error}')
        if error.resp.status == 400:
            print("Error 400: Bad Request. Check the format of start/end times and timezone.")
        elif error.resp.status == 403:
            print("Error 403: Permission denied. Ensure SCOPES include calendar writing permissions (e.g., calendar.events).")
        elif error.resp.status == 404:
            print("Error 404: Calendar not found. Ensure 'primary' calendar exists for the user.")
        return None
    except Exception as e:
        print(f'An unexpected error occurred during calendar event creation: {e}')
        return None




# --- Main Execution Block ---
if __name__ == '__main__':
    # 1. Authenticate and get the API service objects
    # This handles the OAuth flow using credentials.json and token.json for ALL defined SCOPES
    print("Attempting to authenticate with Google APIs (Gmail & Calendar)...")
    # This function now returns two service objects
    gmail_service, calendar_service = authenticate_google_apis()

    # --- Gmail Operations ---
    if gmail_service:
        print("\nAuthentication successful for Gmail. Proceeding with Gmail operations...")

        # # Example: Read recent unread emails (Optional)
        # print("\n--- Reading Emails Example ---")
        # read_emails(service=gmail_service, max_results=3, query='in:inbox is:unread')

        # Example: Send the requested email
        print("\n--- Sending Requested Email ---")
        # !!! IMPORTANT: YOU MUST CHANGE THE RECIPIENT EMAIL BELOW !!!
        recipient_email = "yasser.x.ali@kp.org" # <-- CHANGE THIS TO A VALID EMAIL ADDRESS
        # --- End Important ---
        email_subject = "nba finals - prediction (automated)"
        email_body = "the warriors will win the championship this year\n\n(Sent via Python script)"

        if recipient_email == "your_email@example.com": # Default check
             print("!!! Gmail Sending skipped: Please change 'recipient_email' in the script to a valid address. !!!")
        elif recipient_email == "yasser.x.ali@kp.org": # Specific check
            print(f"Attempting to send email to: {recipient_email}")
            send_result = send_email(recipient_email, email_subject, email_body)
            if send_result:
                print("Gmail send function executed successfully.")
            else:
                print("Gmail send function encountered an error or failed.")
        else:
             print(f"Recipient email '{recipient_email}' is not the expected test address. Sending skipped for safety.")


    else:
        print("\nFailed to authenticate or build Gmail service. Skipping Gmail operations.")

    # --- Calendar Operations ---
    if calendar_service:
        print("\nAuthentication successful for Calendar. Proceeding with Calendar operations...")

        # Example: List upcoming events
        print("\n--- Listing Calendar Events Example ---")
        list_upcoming_events(max_results=5)

        # Example: Create a new calendar event
        print("\n--- Creating Calendar Event Example ---")
        # Define the event details
        event_summary = 'Go to sleep, testing 5:22pm.'
        # Specify the timezone for the event times
        event_timezone = 'America/Los_Angeles' # Use IANA timezone name relevant to you

        try:
            # Get current time in the specified timezone
            tz = pytz.timezone(event_timezone)
            # Create event for tomorrow at 2 PM in the specified timezone
            start_dt = datetime.datetime.now(tz) + datetime.timedelta(days=1)
            start_dt = start_dt.replace(hour=10, minute=0, second=0, microsecond=0)
            # Event duration: 1 hour
            end_dt = start_dt + datetime.timedelta(hours=1)

            # Format dates to RFC3339 string required by the API
            start_iso = start_dt.isoformat() # e.g., '2025-05-05T14:00:00-07:00'
            end_iso = end_dt.isoformat()     # e.g., '2025-05-05T15:00:00-07:00'

            create_result = create_calendar_event(
                service=calendar_service,
                summary=event_summary,
                start_time_str=start_iso,
                end_time_str=end_iso,
                timezone=event_timezone
            )
            if create_result:
                 print("Calendar event creation function executed successfully.")
            else:
                 print("Calendar event creation function encountered an error or failed.")

        except pytz.UnknownTimeZoneError:
             print(f"ERROR: Unknown timezone '{event_timezone}'. Please use a valid IANA timezone name.")
        except Exception as e:
             print(f"An error occurred setting up date/time for calendar event: {e}")

    else:
        print("\nFailed to authenticate or build Calendar service. Skipping Calendar operations.")

    print("\nScript execution finished.")
