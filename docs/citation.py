import requests

# Replace with your paper's DOI
DOI = "10.3389/fnins.2022.999029"

# API endpoint for Semantic Scholar
API_URL = f"https://api.semanticscholar.org/v1/paper/{DOI}"


def fetch_citations():
    response = requests.get(API_URL)
    if response.status_code == 200:
        data = response.json()
        citations = data.get("citations", [])
        return [
            (
                f"* {citation['title']} by {', '.join(author['name'] for author in citation['authors'])} "
                f"({citation['year']})"
            )
            for citation in citations
        ]
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return []


def update_cited_by_file():
    citations = fetch_citations()
    if citations:
        with open("cited_by_list.rst", "w") as file:
            file.write("\n".join(citations))
            print("Updated cited_by_list.rst successfully.")
    else:
        print("No citations found or failed to fetch citations.")


if __name__ == "__main__":
    update_cited_by_file()
