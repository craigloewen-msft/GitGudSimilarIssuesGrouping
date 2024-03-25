from pymongo import MongoClient
import json
import tiktoken
from datetime import datetime
import matplotlib.pyplot as plt
import math

repoShortURL = 'microsoft/powertoys'
startingQueryDate = datetime(2019, 1, 1)
percentBelowTokenNumber = 1024
tokenLengthToExcludeFromView = 1500
allowListEnabled = True
allowListLabels = [
    'Area-Accessibility',
    'Area-App Compat',
    'Area-Bug Report Tool',
    'Area-Build',
    'Area-Context Menu', 'Area-Enterprise', 'Area-Flyout', 'Area-GitHub workflow', 'Area-Localization', 'Area-Logging', 'Area-Module interface', 'Area-OOBE', 'Area-Project Template', 'Area-Quality', 'Area-Runner', 'Area-Setup/Install', 'Area-Telemetry', 'Area-Tests', 'Area-User Interface',
    'FancyZones-Dragging&UI', 'FancyZones-Editor', 'FancyZones-Hotkeys', 'FancyZones-Layouts', 'FancyZones-Settings', 'FancyZones-VirtualDesktop',
    'Idea-Enhancement', 'Issue-Docs', 'Issue-Feature', 'Issue-Translation',
    'Product-Always On Top', 'Product-Awake', 'Product-Color Picker', 'Product-CommandNotFound', 'Product-CropAndLock', 'Product-Display management', 'Product-Environment Variables', 'Product-FancyZones', 'Product-File Explorer', 'Product-File Locksmith', 'Product-Hosts File Editor', 'Product-Image Resizer', 'Product-Keyboard Shortcut Manager', 'Product-Mouse Utilities', 'Product-Mouse Without Borders', 'Product-Paste as plain text', 'Product-Peek', 'Product-Power management', 'Product-PowerRename', 'Product-PowerToys Run', 'Product-Quick Accent', 'Product-Registry Preview', 'Product-Screen Ruler',
    'Product-Settings', 'Product-Shortcut Guide', 'Product-Text Extractor', 'Product-Tweak UI Design', 'Product-Video Conference Mute', 'Product-Virtual Desktop', 'Product-Window Manager',
    'Run-Plugin Manager', 'Run-Plugin', 'Severity-Crash',
    'Issue-Bug'
]


# allowListLabelsWSL = [
#     'ARM','bug','bydesign','console','discussion','distro-mgmt','documentation','enhancement','external','failure-to-launch','feature','file system','fixed-in-wsl2','GPU',
#     'hypervisor-platform','inbox','install','interop','kconfig','kernel','launcher','localization','msix','network','systemd','wsl1','wsl2'
# ]

allowListLabelsString = ','.join(allowListLabels)

# Read configurations from config.json
with open('config.json', 'r') as f:
    config = json.load(f)

# Step 1: Connect to MongoDB and retrieve the repository ID and issue list


def get_repository_id_and_issue_list(repo_name, inStartingQueryDate):
    client = MongoClient(config['devMongoDBConnectionString'])
    db = client['GithubIssueManagement']

    print("Getting repo id and issue list")

    # Retrieve repository ID
    repo_collection = db['repoInfo']
    repo_result = repo_collection.find_one({'shortURL': repo_name})
    repo_id = repo_result['_id']

    # Retrieve issues created more than 3 years ago for the given repository
    issue_collection = db['issueInfo']
    issues = issue_collection.find({
        'repoRef': repo_id,
        'created_at': {'$gt': inStartingQueryDate}
    })

    # Change issue_list to a dictionary
    issue_dict = {str(issue['_id']): issue for issue in issues}

    return repo_id, issue_dict

# Credit to stack overflow https://stackoverflow.com/questions/75804599/openai-api-how-do-i-count-tokens-before-i-send-an-api-request


def getTokenCountFromString(string, encoding_name):
    """
    Returns the number of tokens in a text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def addTokenCountToIssues(issueList):

    for issueID in issueList:
        issue = issueList[issueID]

        try:
            tokenLength = getTokenCountFromString(
                '===Labels to choose from===\n' + allowListLabelsString + '\n\n'
                '===Issue Details===\n' +
                '# ' + issue['title'] + '\n' + issue['body'] + '\n\n'
                + '===Labels to apply===\n',
                "cl100k_base")
        except:
            tokenLength = 0

        issue["tokenLength"] = tokenLength


def getProcessedDict(issueList):

    processedDict = {}

    for issueID in issueList:
        issue = issueList[issueID]
        # For each item in issue['labels'] dictionary, put its name value into a comma separated list
        originalLabelList = [label['name'] for label in issue['labels']]

        # Filter labelList to only have allow listed labels
        if allowListEnabled:
            labelList = list(
                filter(lambda x: x in allowListLabels, originalLabelList))
        else:
            labelList = originalLabelList

        if len(labelList) > 0:
            labelListString = ','.join(labelList)
        else:
            labelListString = '<None>'

        if (issue['tokenLength'] > 0 and issue['tokenLength'] < 1000):
            processedDict[issueID] = {
                "issueDescription": '# ' + issue['title'] + '\n' + issue['body'],
                "labels": labelListString,
                "tokenLength": issue['tokenLength']
            }

    return processedDict


def getPercentBelowTokenLength(issue_list, inTokenLength):
    tokenLengthList = [issue_list[issueID]['tokenLength']
                       for issueID in issue_list]
    tokenLengthList.sort()
    percentBelowTokenValue = 0
    for tokenValue in tokenLengthList:
        if tokenValue <= inTokenLength:
            percentBelowTokenValue += 1
        else:
            break

    percentileIndex = round(percentBelowTokenValue *
                            1.0 / len(tokenLengthList) * 100, 2)

    return percentileIndex


def showHistogramPlotOfTokenLength(issue_list):
    # ignore any tokens above tokenLengthToExcludeFromView
    tokenLengthList = [issue_list[issueID]['tokenLength']
                       for issueID in issue_list if issue_list[issueID]['tokenLength'] < tokenLengthToExcludeFromView]
    numberOfIgnored = len([issue_list[issueID]['tokenLength']
                          for issueID in issue_list if issue_list[issueID]['tokenLength'] >= tokenLengthToExcludeFromView])
    print("Number of tokens ignored: ", numberOfIgnored)
    plt.hist(tokenLengthList, bins=20)
    plt.show()


def exportToJSON(processedDict):
    with open('dataset-classification.json', 'w') as f:
        for entry in processedDict.values():
            f.write(json.dumps(entry) + '\n')

def exportToPrettyJSON(processedDict):
    formattedObject = [processedDict[issueID] for issueID in processedDict]
    with open('issue_data.json', 'w') as f:
        f.write(json.dumps(formattedObject, indent=4))

# Main function
def main():
    repository_name = repoShortURL
    repository_id, issue_list = get_repository_id_and_issue_list(
        repository_name, startingQueryDate)
    addTokenCountToIssues(issue_list)
    processedDict = getProcessedDict(issue_list)
    print("Percent below: ", getPercentBelowTokenLength(
        issue_list, percentBelowTokenNumber))
    showHistogramPlotOfTokenLength(issue_list)
    exportToJSON(processedDict)

def getTrainingIssues():
    repository_name = repoShortURL
    issueQueryDate = datetime(2024, 3, 10)
    repository_id, issue_list = get_repository_id_and_issue_list(
        repository_name, issueQueryDate)
    addTokenCountToIssues(issue_list)
    processedDict = getProcessedDict(issue_list)
    exportToPrettyJSON(processedDict)

if __name__ == "__main__":
    main()
    getTrainingIssues()
