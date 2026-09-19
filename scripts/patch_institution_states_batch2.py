"""
scripts/patch_institution_states_batch2.py

One-time patch for the 225 Institution nodes that had null i.state after batch 1.
Covers all non-cooperative-bank entity types: NBFCs, private/public banks,
payment banks, RRBs, SFBs, PPIs, PAs, CICs, ARCs.

Foreign entities are tagged state="Foreign".
The multi-institution joint-order node (Sodexo + 5 others) is skipped — no single state.
Cooperatieve Rabobank U.A is skipped — intentional null (Dutch bank).

Usage (from rbi_intel/ directory with venv active):
    python scripts/patch_institution_states_batch2.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# ── Setup ─────────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent / ".env")

NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USERNAME"]
NEO4J_PASS = os.environ["NEO4J_PASSWORD"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


# ── State mapping ──────────────────────────────────────────────────────────────
# Keys must match i.name in Neo4j exactly (including whitespace quirks).
# "Foreign" is used for non-Indian entities.
# Entries are omitted where state cannot be determined (Rabobank, joint order).

INSTITUTION_STATES: dict[str, str] = {

    # ── ARC ───────────────────────────────────────────────────────────────────
    # Pan-India ARC → National
    "Phoenix ARC Private Limited": "National",

    # ── Bank-Payment ──────────────────────────────────────────────────────────
    # Rule: banks with no state in name → "National"; foreign → "Foreign"
    "American Express Banking Corp":    "Foreign",
    "Fino Payments Bank Limited":       "National",
    "India Post Payments Bank Limited": "National",
    "Jio Payments Bank Limited":        "National",
    "Paytm Payments Bank Limited":      "National",

    # ── Bank-Private (foreign) ────────────────────────────────────────────────
    "Authorised Dealer Bank – Bank of America, N.A":          "Foreign",
    "Authorised Dealer Bank – The HSBC Limited":              "Foreign",
    "BNP Paribas":                                            "Foreign",
    "Bank of Bahrain & Kuwait BSC, India Operations":         "Foreign",
    "Citibank N.A":                                           "Foreign",
    "Deutsche Bank AG":                                       "Foreign",
    "Industrial Bank of Korea":                               "Foreign",
    "KEB Hana Bank":                                          "Foreign",
    "MUFG Bank Ltd":                                          "Foreign",
    "Standard Chartered Bank – India":                        "Foreign",
    "Standard Chartered Bank-India":                          "Foreign",
    "The Hongkong and Shanghai Banking Corporation Limited":  "Foreign",
    "Woori Bank":                                             "Foreign",

    # ── Bank-Private (Indian, no state in name → "National") ─────────────────
    "Authorised Dealer Bank – HDFC Bank Ltd": "National",
    "Axis Bank Limited":                      "National",
    "Bandhan Bank Limited":                   "National",
    "CSB Bank Limited":                       "National",
    "City Union Bank Limited":                "National",
    "DCB Bank Limited":                       "National",
    "Dhanlaxmi Bank Ltd":                     "National",
    "Federal Bank Limited":                   "National",
    "HDFC Bank Limited":                      "National",
    "ICICI Bank Ltd":                         "National",
    "IDBI Bank Limited":                      "National",
    "IDFC First Bank Limited":                "National",
    "IndusInd Bank Ltd":                      "National",
    "Jammu and Kashmir Bank Limited":         "National",
    "Karnataka Bank Limited":                 "National",
    "Karur Vysya Bank Limited":               "National",
    "Kotak Mahindra Bank Limited":            "National",
    "RBL Bank Limited":                       "National",
    "SBM Bank (India) Limited":               "National",
    "South Indian Bank Limited":              "National",
    "Tamilnad Mercantile Bank Limited":       "National",
    "The Nainital Bank Limited":              "National",
    "Yes Bank Limited":                       "National",

    # ── Bank-Private (cooperative banks misclassified — state in name) ────────
    "Balitikuri Cooperative  Bank Ltd., Balitikuri, West Bengal":                                "West Bengal",
    "Chikkamagaluru District Cooperative Central Bank Ltd., Chikkamagaluru, Karnataka":          "Karnataka",
    "Innespeta Cooperative Urban Bank Ltd., Rajahmundry, East Godavari District, Andhra Pradesh":"Andhra Pradesh",
    "Jowai Cooperative Urban Bank Limited, Meghalaya":                                          "Meghalaya",
    "Mangaldai Nagar Samabai Bank Limited, Mangaldai (Assam)":                                  "Assam",
    "Mizoram Urban Cooperative Development Bank Limited, Aizawl":                               "Mizoram",
    "Raigad District Central Co-op Bank Ltd, Maharashtra":                                      "Maharashtra",
    "Samata Cooperative Development Bank Ltd., Kolkata (West Bengal)":                          "West Bengal",
    "Telangana State Cooperative Apex Bank Limited, Hyderabad":                                 "Telangana",
    "The Balasore Cooperative Urban Bank Limited, Balasore, Odisha":                            "Odisha",
    "The Berhampur Cooperative Urban Bank Ltd., Odisha":                                        "Odisha",
    "The Citizens Urban Cooperative  Bank Ltd., Jalandhar":                                     "Punjab",
    "The Delhi State Cooperative  Bank Ltd., New Delhi":                                        "Delhi",
    "The Guntur Women Cooperative Urban Bank Ltd., Guntur, Andhra Pradesh":                     "Andhra Pradesh",
    "The Krishna District Cooperative Central Bank Ltd., Machilipatnam, Krishna District, Andhra Pradesh": "Andhra Pradesh",
    "The Nainital Bank Ltd., Uttarakhand":                                                       "Uttarakhand",
    "The Sonepat Central Cooperative  Bank Ltd., Haryana":                                      "Haryana",
    "The Swasakthi Mercantile Cooperative Urban Bank Ltd., Vijayawada, Krishna District, Andhra Pradesh": "Andhra Pradesh",
    "the Delhi Nagrik Sehkari Bank Limited, New Delhi":                                         "Delhi",

    # ── Bank-Public ───────────────────────────────────────────────────────────
    # Rule: all Indian PSBs → "National"; Sonali Bank (Bangladesh) → "Foreign"
    "Bank of Baroda":       "National",
    "Bank of India":        "National",
    "Bank of Maharashtra":  "National",
    "Canara Bank":          "National",
    "Central Bank of India":"National",
    "Indian Bank":          "National",
    "Indian Overseas Bank": "National",
    "Punjab & Sind Bank":   "National",
    "Punjab National Bank": "National",
    "Punjab and Sind Bank": "National",
    "Sonali Bank PLC":      "Foreign",
    "State Bank of India":  "National",
    "UCO Bank":             "National",
    "Union Bank of India":  "National",

    # ── Bank-RRB ──────────────────────────────────────────────────────────────
    # State present in name → extract; no state in name → "National" per bank rule
    "Andhra Pragathi Grameena Bank, Andhra Pradesh": "Andhra Pradesh",
    "Arunachal Pradesh Rural Bank":                  "Arunachal Pradesh",
    "Baroda Gujarat Gramin Bank, Vadodara (Gujarat)":"Gujarat",
    "Karnataka Gramin Bank, Karnataka":              "Karnataka",
    "Maharashtra Gramin Bank, Aurangabad":           "Maharashtra",
    "Manipur Rural Bank":                            "Manipur",
    "Mizoram Rural Bank":                            "Mizoram",
    "Odisha Gramya Bank, Odisha":                    "Odisha",
    "Punjab Gramin Bank, Kapurthala, Punjab":        "Punjab",
    "The Uttarakhand Grameen Bank, Dehradun":        "Uttarakhand",
    "Tripura Gramin Bank":                           "Tripura",

    # ── Bank-SFB ──────────────────────────────────────────────────────────────
    # Rule: banks with no state in name → "National"
    "ESAF Small Finance Bank Limited":       "National",
    "Equitas Small Finance Bank Limited":    "National",
    "Jana Small Finance Bank Limited":       "National",
    "North East Small Finance Bank Limited": "National",
    "Suryoday Small Finance Bank Ltd":       "National",
    "Ujjivan Small Finance Bank Limited":    "National",

    # ── CIC ───────────────────────────────────────────────────────────────────
    # Mumbai-headquartered credit bureaus → Maharashtra; one misclassified coop bank
    "CRIF High Mark Credit Information Services Private Limited, Mumbai": "Maharashtra",
    "Equifax Credit Information Services Private Limited, Mumbai":        "Maharashtra",
    "Experian Credit Information Company of India Private Limited":       "Maharashtra",
    "Nashik Zilla Girna Sahakari Bank Ltd., Nashik":                     "Maharashtra",
    "TransUnion CIBIL Limited, Mumbai":                                  "Maharashtra",

    # ── NBFC-HFC ──────────────────────────────────────────────────────────────
    # State in name → extract; pan-India HFCs → "National"; regional → state
    "Aadhar Housing Finance Limited":                            "National",
    "Altum Credo Home Finance Private Limited, Pune, Maharashtra":"Maharashtra",
    "Bajaj Housing Finance Limited, Pune":                       "Maharashtra",
    "Cent Bank Home Finance Limited":                            "Madhya Pradesh",
    "Family Home Finance Private Limited, Mumbai, Maharashtra":  "Maharashtra",
    "Godrej Housing Finance Limited":                            "Maharashtra",
    "HDB Financial Services Ltd":                                "National",
    "Hinduja Leyland Finance Limited":                           "Tamil Nadu",
    "Housing Development Finance Corporation Limited, Mumbai":   "National",
    "Housing and Urban Development Corporation Limited":         "National",
    "India Home Loan Limited, Mumbai Maharashtra":               "Maharashtra",
    "India Home Loan Ltd., Mumbai, Maharashtra":                 "Maharashtra",
    "JM Financial Home Loans Limited":                           "National",
    "Khush Housing Finance Private Limited, Mumbai, Maharashtra":"Maharashtra",
    "LIC Housing Finance Limited":                               "National",
    "Mahindra Rural Housing Finance Limited":                    "National",
    "Muthoot FinCorp Limited":                                   "Kerala",
    "Nido Home Finance Limited":                                 "Maharashtra",
    "Sahara Housingfina Corporation Limited, West Bengal":       "West Bengal",
    "Sundaram Home Finance Limited":                             "Tamil Nadu",
    "Truhome Finance Limited":                                   "Tamil Nadu",
    "Viva Home Finance Limited, Palghar, Maharashtra":           "Maharashtra",
    "West End Housing Finance Limited, Mumbai, Maharashtra":     "Maharashtra",

    # ── NBFC-ICC ──────────────────────────────────────────────────────────────
    # State in name → extract; pan-India NBFCs → "National"; regional → state
    "Amrit Malwa Capital Limited, Jalandhar, Punjab":                    "Punjab",
    "Anand Rathi Global Finance Limited, Mumbai, Maharashtra":           "Maharashtra",
    "Aptus Finance India Private Limited, Chennai":                      "Tamil Nadu",
    "Ayodhya Finlease Limited":                                          "Uttar Pradesh",
    "Bahadur Chand Investments Private Limited, New Delhi":              "Delhi",
    "Bajaj Finance Limited":                                             "National",
    "Balaji Instalments Limited":                                        "Tamil Nadu",
    "Berar Finance Limited, Nagpur, Maharashtra":                        "Maharashtra",
    "Daimler Financial Services India Private Limited, Pune, Maharashtra":"Maharashtra",
    "Datson Exports Ltd., West Bengal":                                  "West Bengal",
    "Dhani Loans and Services Limited,  Gurugram, Haryana":              "Haryana",
    "ECL Finance Limited, Mumbai, Maharashtra":                          "Maharashtra",
    "Early Salary Services Private Limited, Pune, Maharashtra":          "Maharashtra",
    "Fedbank Financial Services Limited":                                "Kerala",
    "Finquest Financial Solutions Private Limited, Mumbai, Maharashtra": "Maharashtra",
    "GPT Sons Private Limited":                                          "West Bengal",
    "Glowmore Finance Private Limited, Odisha":                          "Odisha",
    "GoCapital Finance Limited, Chennai, Tamil Nadu":                    "Tamil Nadu",
    "Hero FinCorp Limited":                                              "National",
    "Hewlett Packard Financial Services (India) Private Limited":        "Karnataka",
    "Hewlett-Packard Financial Services (India) Private Ltd":            "Karnataka",
    "IGH Holdings Private Limited, Mumbai":                              "Maharashtra",
    "IIFL Finance Limited":                                              "National",
    "IIFL Samasta Finance Limited":                                      "Karnataka",
    "Indiabulls Commercial  Credit Limited, New Delhi":                  "Delhi",
    "Indian School Finance Company Pvt. Ltd., Telangana":                "Telangana",
    "Indostar Capital Finance Limited, Mumbai, Maharashtra":             "Maharashtra",
    "JM Financial Products Limited":                                     "National",
    "Jumbo Finvest (India) Limited":                                     "Maharashtra",
    "Jupiter Capital Private Limited, Bengaluru":                        "Karnataka",
    "KKR India Financial Services Limited, Mumbai":                      "Maharashtra",
    "KLM Axiva Finvest Limited":                                         "Kerala",
    "Keertana Finserv Limited":                                          "Andhra Pradesh",
    "Kosamattam Finance Limited":                                        "Kerala",
    "Krazybee Services Private Limited, Bangalore":                      "Karnataka",
    "L&T Finance Limited, Mumbai, Maharashtra":                          "Maharashtra",
    "M.L. Securities and Finance Private Limited, Indore (M.P)":        "Madhya Pradesh",
    "Maanaveeya Development & Finance  Private Limited":                 "Telangana",
    "Mahindra & Mahindra Financial Services Limited":                    "National",
    "Manappuram Finance Limited":                                        "Kerala",
    "Mannakrishna Investments Private Limited":                          "Kerala",
    "Maxvalue Credits and Investments Limited, Thrissur, Kerala":        "Kerala",
    "Mercedes-Benz Financial Services India Private Limited (formerly known as Daimler Financial Services India Private Limited), Pune, Maharashtra": "Maharashtra",
    "Mikhael Capitalize Pvt. Ltd., Kerala":                              "Kerala",
    "Mulamoottil Financiers Limited, Kozhencherry, Pathanamthitta District, Kerala": "Kerala",
    "Muthoot Finance Limited":                                           "Kerala",
    "Muthoot Money Limited, Ernakulam, Kerala":                         "Kerala",
    "NABFINS Limited, Bengaluru":                                        "Karnataka",
    "Nissan Renault Financial Services India Private Limited":           "Tamil Nadu",
    "Northern Arc Capital Limited":                                      "National",
    "Nuvama Wealth Finance Limited (formerly known as Edelweiss Finance and Investments Limited), Mumbai": "National",
    "Ocean Capital Market Limited, Rourkela, Odisha":                   "Odisha",
    "PayMe India Financial Services Private Limited":                    "Delhi",
    "Poonawalla Fincorp Limited,  Pune, Maharashtra":                    "Maharashtra",
    "Pooram Finserv Pvt Limited, Thrissur, Kerala":                      "Kerala",
    "Power Finance Corporation Limited":                                 "National",
    "Purvaja Fincap Pvt. Ltd., Tamil Nadu":                              "Tamil Nadu",
    "Rain Bow Finance India Ltd., Chennai, Tamil Nadu":                  "Tamil Nadu",
    "SG Finserve limited":                                               "Maharashtra",
    "SMFG India Credit Company Limited":                                 "National",
    "Sakthi Finance Limited, Coimbatore, Tamil Nadu":                    "Tamil Nadu",
    "Sankhya Financial Services Private Limited, Mumbai":                "Maharashtra",
    "Sappers Finance and Consultancy Private Limited, Kolkata":          "West Bengal",
    "Savery Transport Finance Ltd., Chennai, Tamil Nadu":                "Tamil Nadu",
    "Seyad Shariat Finance Limited, Tirunelveli, Tamil Nadu":            "Tamil Nadu",
    "Shaha Finlease Private Limited, Mumbai":                            "Maharashtra",
    "Shriram City Union Finance Limited":                                "Tamil Nadu",
    "Shriram Finance Limited":                                           "Tamil Nadu",
    "Sonata Finance Private Limited":                                    "Uttar Pradesh",
    "Spandana Sphoorty  Financial Limited, Hyderabad":                   "Telangana",
    "Srestha Finvest Limited, Chennai, Tamil Nadu":                      "Tamil Nadu",
    "Swagatam Marketing Private Limited, Kolkata":                       "West Bengal",
    "UCA Finvest Private Limited, New Delhi":                            "Delhi",
    "VSJ Investments Private Limited,  Mumbai, Maharashtra":             "Maharashtra",
    "Valuecorp Securities & Finance Limited":                            "Maharashtra",
    "Vanchinad Finance Pvt. Ltd., Ernakulam, Kerala":                    "Kerala",
    "Vinayaka Capsec Private Limited":                                   "Karnataka",

    # ── NBFC-MFI ──────────────────────────────────────────────────────────────
    "Asirvad Micro Finance Limited":              "Kerala",
    "Belstar Microfinance Limited, Chennai":      "Tamil Nadu",
    "Dhani Loans and Services Limited, Gurugram": "Haryana",
    "Navi Finserv Limited":                       "Karnataka",

    # ── NBFC-P2P ──────────────────────────────────────────────────────────────
    "Bansal Credits Limited":                                       "Delhi",
    'Bridge Fintech Solutions Private Limited ("Finzy")':           "Maharashtra",
    "Fairassets Technologies India Private Limited ('Faircent')":   "Haryana",
    'Innofin Solutions Private Limited  ("LenDen Club")':           "Maharashtra",
    "NDX P2P Private Limited ('LiquiLoans')":                       "Maharashtra",
    "Pinnacle Capital Solutions Private Limited, Jharkhand":        "Jharkhand",
    'Rang De P2P Financial Services Limited ("Rang De")':           "Karnataka",
    "Transactree Technologies Private Limited ('Lendbox')":         "Delhi",
    "Village Financial Services Limited, Kolkata":                  "West Bengal",
    "Visionary Financepeer Private Limited":                        "Maharashtra",

    # ── PA ────────────────────────────────────────────────────────────────────
    "Cashfree Payments India Private Limited":         "Karnataka",
    "Pinnacle Capital Solutions Pvt. Ltd., Jharkhand": "Jharkhand",

    # ── PPI ───────────────────────────────────────────────────────────────────
    # Joint order node skipped — no single state assignable
    "Amazon Pay (India) Private Limited":                          "Karnataka",
    "LivQuik Technology (India) Private Limited":                  "Maharashtra",
    "Mynd Solutions Private Limited":                              "Haryana",
    "Obopay Mobile Technology India Private Limited":              "Karnataka",
    "Ola Financial Services Private Limited":                      "Karnataka",
    "Paul Merchants Finance Private Limited":                      "Punjab",
    "PhonePe Limited (formerly known as PhonePe Private Limited)": "Karnataka",
    "Spice Money Limited":                                         "Delhi",
    "Unimoni Financial Services Limited":                          "Kerala",
    "Vakrangee Limited":                                           "Maharashtra",
}


# ── Neo4j patch ────────────────────────────────────────────────────────────────

def patch_state(driver, name: str, state: str) -> int:
    query = """
    MATCH (i:Institution {name: $name})
    SET i.state = $state
    RETURN count(i) AS updated
    """
    with driver.session() as s:
        result = s.run(query, name=name, state=state)
        record = result.single()
        return record["updated"] if record else 0


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    total     = len(INSTITUTION_STATES)
    patched   = 0
    unmatched = []

    log.info("Patching %d institutions...", total)

    for name, state in INSTITUTION_STATES.items():
        updated = patch_state(driver, name, state)
        if updated > 0:
            patched += 1
        else:
            unmatched.append(name)

    driver.close()

    log.info("")
    log.info("═" * 60)
    log.info("PATCH COMPLETE")
    log.info("  Mapping entries : %d", total)
    log.info("  Patched         : %d", patched)
    log.info("  Unmatched       : %d", len(unmatched))

    if unmatched:
        log.info("")
        log.info("Unmatched (no Institution node found — possible name mismatch):")
        for name in unmatched:
            log.info("  - %s", name)

    log.info("")
    log.info("Verify with:")
    log.info("  MATCH (i:Institution) WHERE i.state IS NULL RETURN i.entity_type, COUNT(i) ORDER BY count DESC")


if __name__ == "__main__":
    main()
