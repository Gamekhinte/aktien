# ============================================================
# 📊 Crypto & Stock Pattern Analyzer – app.py
# ============================================================
# requirements.txt (Streamlit Cloud):
#   streamlit>=1.37   # >=1.37 wird für die automatische Live-Scan-Aktualisierung (st.fragment) benötigt
#   yfinance
#   pandas
#   numpy
#   google-genai
#   plotly
#   scikit-learn
#   supabase
#
# Start lokal: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import io
import os
import json
import itertools
import hashlib
import urllib.request
import urllib.error
from datetime import datetime, time
from zoneinfo import ZoneInfo

try:
    from broker_feed import IBKRFeed  # type: ignore[import-not-found]
except ImportError:
    IBKRFeed = None

try:
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from supabase import create_client, Client
    SUPABASE_PACKAGE_AVAILABLE = True
except ImportError:
    create_client = None
    Client = None
    SUPABASE_PACKAGE_AVAILABLE = False

# ------------------------------------------------------------
# Persistentes gemeinsames Lernsystem (Supabase)
# ------------------------------------------------------------
LEARNING_FEATURE_VERSION = "v1"
AUTO_TRAIN_MIN_NEW_EXAMPLES = 100
MODEL_ALGORITHM = "GradientBoostingClassifier"
TRAINING_MIN_SAMPLES = 200
SCANNER_ACCOUNT_KEY = "shared_scanner_v1"
SCANNER_UNIVERSE: dict[str, str] = {}
# --- ETFs ---
SCANNER_UNIVERSE["S&P 500 ETF"] = "SPY"
SCANNER_UNIVERSE["Nasdaq 100 ETF"] = "QQQ"
SCANNER_UNIVERSE["Dow Jones ETF"] = "DIA"
SCANNER_UNIVERSE["Russell 2000 ETF"] = "IWM"
SCANNER_UNIVERSE["Gold ETF"] = "GLD"
SCANNER_UNIVERSE["Silber ETF"] = "SLV"
SCANNER_UNIVERSE["Total US Market ETF"] = "VTI"
SCANNER_UNIVERSE["Vanguard S&P 500 ETF"] = "VOO"
# --- Kryptowaehrungen (Top ~75 nach Marktkapitalisierung) ---
SCANNER_UNIVERSE["Bitcoin"] = "BTC-USD"
SCANNER_UNIVERSE["Ethereum"] = "ETH-USD"
SCANNER_UNIVERSE["BNB"] = "BNB-USD"
SCANNER_UNIVERSE["Solana"] = "SOL-USD"
SCANNER_UNIVERSE["XRP"] = "XRP-USD"
SCANNER_UNIVERSE["Cardano"] = "ADA-USD"
SCANNER_UNIVERSE["Dogecoin"] = "DOGE-USD"
SCANNER_UNIVERSE["Avalanche"] = "AVAX-USD"
SCANNER_UNIVERSE["TRON"] = "TRX-USD"
SCANNER_UNIVERSE["Chainlink"] = "LINK-USD"
SCANNER_UNIVERSE["Polkadot"] = "DOT-USD"
SCANNER_UNIVERSE["Polygon"] = "MATIC-USD"
SCANNER_UNIVERSE["Shiba Inu"] = "SHIB-USD"
SCANNER_UNIVERSE["Litecoin"] = "LTC-USD"
SCANNER_UNIVERSE["Bitcoin Cash"] = "BCH-USD"
SCANNER_UNIVERSE["NEAR Protocol"] = "NEAR-USD"
SCANNER_UNIVERSE["Uniswap"] = "UNI-USD"
SCANNER_UNIVERSE["Internet Computer"] = "ICP-USD"
SCANNER_UNIVERSE["Stellar"] = "XLM-USD"
SCANNER_UNIVERSE["Ethereum Classic"] = "ETC-USD"
SCANNER_UNIVERSE["Filecoin"] = "FIL-USD"
SCANNER_UNIVERSE["Cosmos"] = "ATOM-USD"
SCANNER_UNIVERSE["Hedera"] = "HBAR-USD"
SCANNER_UNIVERSE["VeChain"] = "VET-USD"
SCANNER_UNIVERSE["Optimism"] = "OP-USD"
SCANNER_UNIVERSE["Maker"] = "MKR-USD"
SCANNER_UNIVERSE["The Graph"] = "GRT-USD"
SCANNER_UNIVERSE["Algorand"] = "ALGO-USD"
SCANNER_UNIVERSE["Aave"] = "AAVE-USD"
SCANNER_UNIVERSE["Quant"] = "QNT-USD"
SCANNER_UNIVERSE["MultiversX"] = "EGLD-USD"
SCANNER_UNIVERSE["The Sandbox"] = "SAND-USD"
SCANNER_UNIVERSE["Decentraland"] = "MANA-USD"
SCANNER_UNIVERSE["Tezos"] = "XTZ-USD"
SCANNER_UNIVERSE["Theta Network"] = "THETA-USD"
SCANNER_UNIVERSE["EOS"] = "EOS-USD"
SCANNER_UNIVERSE["Flow"] = "FLOW-USD"
SCANNER_UNIVERSE["Chiliz"] = "CHZ-USD"
SCANNER_UNIVERSE["Kava"] = "KAVA-USD"
SCANNER_UNIVERSE["Monero"] = "XMR-USD"
SCANNER_UNIVERSE["Cronos"] = "CRO-USD"
SCANNER_UNIVERSE["THORChain"] = "RUNE-USD"
SCANNER_UNIVERSE["Fantom"] = "FTM-USD"
SCANNER_UNIVERSE["Injective"] = "INJ-USD"
SCANNER_UNIVERSE["Render"] = "RNDR-USD"
SCANNER_UNIVERSE["dogwifhat"] = "WIF-USD"
SCANNER_UNIVERSE["Celestia"] = "TIA-USD"
SCANNER_UNIVERSE["Sei"] = "SEI-USD"
SCANNER_UNIVERSE["Kaspa"] = "KAS-USD"
SCANNER_UNIVERSE["Bonk"] = "BONK-USD"
SCANNER_UNIVERSE["Jupiter"] = "JUP-USD"
SCANNER_UNIVERSE["Pyth Network"] = "PYTH-USD"
SCANNER_UNIVERSE["Ethena"] = "ENA-USD"
SCANNER_UNIVERSE["Ondo"] = "ONDO-USD"
SCANNER_UNIVERSE["JasmyCoin"] = "JASMY-USD"
SCANNER_UNIVERSE["Gala"] = "GALA-USD"
SCANNER_UNIVERSE["Mina Protocol"] = "MINA-USD"
SCANNER_UNIVERSE["Arweave"] = "AR-USD"
SCANNER_UNIVERSE["Oasis Network"] = "ROSE-USD"
SCANNER_UNIVERSE["Zilliqa"] = "ZIL-USD"
SCANNER_UNIVERSE["Ankr"] = "ANKR-USD"
SCANNER_UNIVERSE["Enjin Coin"] = "ENJ-USD"
SCANNER_UNIVERSE["Basic Attention Token"] = "BAT-USD"
SCANNER_UNIVERSE["0x Protocol"] = "ZRX-USD"
SCANNER_UNIVERSE["Compound"] = "COMP-USD"
SCANNER_UNIVERSE["Synthetix"] = "SNX-USD"
SCANNER_UNIVERSE["yearn.finance"] = "YFI-USD"
SCANNER_UNIVERSE["Curve DAO"] = "CRV-USD"
SCANNER_UNIVERSE["Lido DAO"] = "LDO-USD"
SCANNER_UNIVERSE["dYdX"] = "DYDX-USD"
SCANNER_UNIVERSE["GMX"] = "GMX-USD"
SCANNER_UNIVERSE["1inch"] = "1INCH-USD"
SCANNER_UNIVERSE["SushiSwap"] = "SUSHI-USD"
SCANNER_UNIVERSE["Convex Finance"] = "CVX-USD"
SCANNER_UNIVERSE["Kusama"] = "KSM-USD"
SCANNER_UNIVERSE["Waves"] = "WAVES-USD"
SCANNER_UNIVERSE["IOTA"] = "IOTA-USD"
SCANNER_UNIVERSE["NEO"] = "NEO-USD"
SCANNER_UNIVERSE["Dash"] = "DASH-USD"
SCANNER_UNIVERSE["Zcash"] = "ZEC-USD"
SCANNER_UNIVERSE["Qtum"] = "QTUM-USD"
# --- S&P 500 Aktien (alle aktuellen Mitglieder) ---
SCANNER_UNIVERSE["Agilent Technologies"] = "A"
SCANNER_UNIVERSE["Apple Inc."] = "AAPL"
SCANNER_UNIVERSE["AbbVie"] = "ABBV"
SCANNER_UNIVERSE["Airbnb"] = "ABNB"
SCANNER_UNIVERSE["Abbott Laboratories"] = "ABT"
SCANNER_UNIVERSE["Arch Capital Group"] = "ACGL"
SCANNER_UNIVERSE["Accenture"] = "ACN"
SCANNER_UNIVERSE["Adobe Inc."] = "ADBE"
SCANNER_UNIVERSE["Analog Devices"] = "ADI"
SCANNER_UNIVERSE["Archer Daniels Midland"] = "ADM"
SCANNER_UNIVERSE["Automatic Data Processing"] = "ADP"
SCANNER_UNIVERSE["Autodesk"] = "ADSK"
SCANNER_UNIVERSE["Ameren"] = "AEE"
SCANNER_UNIVERSE["American Electric Power"] = "AEP"
SCANNER_UNIVERSE["AES Corporation"] = "AES"
SCANNER_UNIVERSE["Aflac"] = "AFL"
SCANNER_UNIVERSE["American International Group"] = "AIG"
SCANNER_UNIVERSE["Assurant"] = "AIZ"
SCANNER_UNIVERSE["Arthur J. Gallagher & Co."] = "AJG"
SCANNER_UNIVERSE["Akamai Technologies"] = "AKAM"
SCANNER_UNIVERSE["Albemarle Corporation"] = "ALB"
SCANNER_UNIVERSE["Align Technology"] = "ALGN"
SCANNER_UNIVERSE["Allstate"] = "ALL"
SCANNER_UNIVERSE["Allegion"] = "ALLE"
SCANNER_UNIVERSE["Applied Materials"] = "AMAT"
SCANNER_UNIVERSE["Amcor"] = "AMCR"
SCANNER_UNIVERSE["Advanced Micro Devices"] = "AMD"
SCANNER_UNIVERSE["Ametek"] = "AME"
SCANNER_UNIVERSE["Amgen"] = "AMGN"
SCANNER_UNIVERSE["Ameriprise Financial"] = "AMP"
SCANNER_UNIVERSE["American Tower"] = "AMT"
SCANNER_UNIVERSE["Amazon"] = "AMZN"
SCANNER_UNIVERSE["Arista Networks"] = "ANET"
SCANNER_UNIVERSE["Aon plc"] = "AON"
SCANNER_UNIVERSE["A. O. Smith"] = "AOS"
SCANNER_UNIVERSE["APA Corporation"] = "APA"
SCANNER_UNIVERSE["Air Products"] = "APD"
SCANNER_UNIVERSE["Amphenol"] = "APH"
SCANNER_UNIVERSE["Apollo Global Management"] = "APO"
SCANNER_UNIVERSE["AppLovin"] = "APP"
SCANNER_UNIVERSE["Aptiv"] = "APTV"
SCANNER_UNIVERSE["Alexandria Real Estate Equities"] = "ARE"
SCANNER_UNIVERSE["Ares Management"] = "ARES"
SCANNER_UNIVERSE["Atmos Energy"] = "ATO"
SCANNER_UNIVERSE["AvalonBay Communities"] = "AVB"
SCANNER_UNIVERSE["Broadcom"] = "AVGO"
SCANNER_UNIVERSE["Avery Dennison"] = "AVY"
SCANNER_UNIVERSE["American Water Works"] = "AWK"
SCANNER_UNIVERSE["Axon Enterprise"] = "AXON"
SCANNER_UNIVERSE["American Express"] = "AXP"
SCANNER_UNIVERSE["AutoZone"] = "AZO"
SCANNER_UNIVERSE["Boeing"] = "BA"
SCANNER_UNIVERSE["Bank of America"] = "BAC"
SCANNER_UNIVERSE["Ball Corporation"] = "BALL"
SCANNER_UNIVERSE["Baxter International"] = "BAX"
SCANNER_UNIVERSE["Best Buy"] = "BBY"
SCANNER_UNIVERSE["Becton Dickinson"] = "BDX"
SCANNER_UNIVERSE["Franklin Resources"] = "BEN"
SCANNER_UNIVERSE["Brown–Forman"] = "BF-B"
SCANNER_UNIVERSE["Bunge Global"] = "BG"
SCANNER_UNIVERSE["Biogen"] = "BIIB"
SCANNER_UNIVERSE["Booking Holdings"] = "BKNG"
SCANNER_UNIVERSE["Baker Hughes"] = "BKR"
SCANNER_UNIVERSE["Builders FirstSource"] = "BLDR"
SCANNER_UNIVERSE["BlackRock"] = "BLK"
SCANNER_UNIVERSE["Bristol Myers Squibb"] = "BMY"
SCANNER_UNIVERSE["BNY Mellon"] = "BNY"
SCANNER_UNIVERSE["Broadridge Financial Solutions"] = "BR"
SCANNER_UNIVERSE["Berkshire Hathaway"] = "BRK-B"
SCANNER_UNIVERSE["Brown & Brown"] = "BRO"
SCANNER_UNIVERSE["Boston Scientific"] = "BSX"
SCANNER_UNIVERSE["Blackstone Inc."] = "BX"
SCANNER_UNIVERSE["BXP, Inc."] = "BXP"
SCANNER_UNIVERSE["Citigroup"] = "C"
SCANNER_UNIVERSE["Cardinal Health"] = "CAH"
SCANNER_UNIVERSE["Carrier Global"] = "CARR"
SCANNER_UNIVERSE["Casey's"] = "CASY"
SCANNER_UNIVERSE["Caterpillar Inc."] = "CAT"
SCANNER_UNIVERSE["Chubb Limited"] = "CB"
SCANNER_UNIVERSE["Cboe Global Markets"] = "CBOE"
SCANNER_UNIVERSE["CBRE Group"] = "CBRE"
SCANNER_UNIVERSE["Crown Castle"] = "CCI"
SCANNER_UNIVERSE["Carnival Corporation"] = "CCL"
SCANNER_UNIVERSE["Cadence Design Systems"] = "CDNS"
SCANNER_UNIVERSE["CDW Corporation"] = "CDW"
SCANNER_UNIVERSE["Constellation Energy"] = "CEG"
SCANNER_UNIVERSE["CF Industries"] = "CF"
SCANNER_UNIVERSE["Citizens Financial Group"] = "CFG"
SCANNER_UNIVERSE["Church & Dwight"] = "CHD"
SCANNER_UNIVERSE["C.H. Robinson"] = "CHRW"
SCANNER_UNIVERSE["Charter Communications"] = "CHTR"
SCANNER_UNIVERSE["Cigna"] = "CI"
SCANNER_UNIVERSE["Ciena"] = "CIEN"
SCANNER_UNIVERSE["Cincinnati Financial"] = "CINF"
SCANNER_UNIVERSE["Colgate-Palmolive"] = "CL"
SCANNER_UNIVERSE["Clorox"] = "CLX"
SCANNER_UNIVERSE["Comcast"] = "CMCSA"
SCANNER_UNIVERSE["CME Group"] = "CME"
SCANNER_UNIVERSE["Chipotle Mexican Grill"] = "CMG"
SCANNER_UNIVERSE["Cummins"] = "CMI"
SCANNER_UNIVERSE["CMS Energy"] = "CMS"
SCANNER_UNIVERSE["Centene Corporation"] = "CNC"
SCANNER_UNIVERSE["CenterPoint Energy"] = "CNP"
SCANNER_UNIVERSE["Capital One"] = "COF"
SCANNER_UNIVERSE["Coherent Corp."] = "COHR"
SCANNER_UNIVERSE["Coinbase"] = "COIN"
SCANNER_UNIVERSE["Cooper Companies (The)"] = "COO"
SCANNER_UNIVERSE["ConocoPhillips"] = "COP"
SCANNER_UNIVERSE["Cencora"] = "COR"
SCANNER_UNIVERSE["Costco"] = "COST"
SCANNER_UNIVERSE["Corpay"] = "CPAY"
SCANNER_UNIVERSE["Copart"] = "CPRT"
SCANNER_UNIVERSE["Camden Property Trust"] = "CPT"
SCANNER_UNIVERSE["CRH plc"] = "CRH"
SCANNER_UNIVERSE["Charles River Laboratories"] = "CRL"
SCANNER_UNIVERSE["Salesforce"] = "CRM"
SCANNER_UNIVERSE["CrowdStrike"] = "CRWD"
SCANNER_UNIVERSE["Cisco"] = "CSCO"
SCANNER_UNIVERSE["CoStar Group"] = "CSGP"
SCANNER_UNIVERSE["CSX Corporation"] = "CSX"
SCANNER_UNIVERSE["Cintas"] = "CTAS"
SCANNER_UNIVERSE["Cognizant"] = "CTSH"
SCANNER_UNIVERSE["Corteva"] = "CTVA"
SCANNER_UNIVERSE["Carvana"] = "CVNA"
SCANNER_UNIVERSE["CVS Health"] = "CVS"
SCANNER_UNIVERSE["Chevron Corporation"] = "CVX"
SCANNER_UNIVERSE["Dominion Energy"] = "D"
SCANNER_UNIVERSE["Delta Air Lines"] = "DAL"
SCANNER_UNIVERSE["DoorDash"] = "DASH"
SCANNER_UNIVERSE["DuPont"] = "DD"
SCANNER_UNIVERSE["Datadog"] = "DDOG"
SCANNER_UNIVERSE["Deere & Company"] = "DE"
SCANNER_UNIVERSE["Deckers Brands"] = "DECK"
SCANNER_UNIVERSE["Dell Technologies"] = "DELL"
SCANNER_UNIVERSE["Dollar General"] = "DG"
SCANNER_UNIVERSE["Quest Diagnostics"] = "DGX"
SCANNER_UNIVERSE["D. R. Horton"] = "DHI"
SCANNER_UNIVERSE["Danaher Corporation"] = "DHR"
SCANNER_UNIVERSE["Walt Disney Company (The)"] = "DIS"
SCANNER_UNIVERSE["Digital Realty"] = "DLR"
SCANNER_UNIVERSE["Dollar Tree"] = "DLTR"
SCANNER_UNIVERSE["Healthpeak Properties"] = "DOC"
SCANNER_UNIVERSE["Dover Corporation"] = "DOV"
SCANNER_UNIVERSE["Dow Inc."] = "DOW"
SCANNER_UNIVERSE["Domino's"] = "DPZ"
SCANNER_UNIVERSE["Darden Restaurants"] = "DRI"
SCANNER_UNIVERSE["DTE Energy"] = "DTE"
SCANNER_UNIVERSE["Duke Energy"] = "DUK"
SCANNER_UNIVERSE["DaVita"] = "DVA"
SCANNER_UNIVERSE["Devon Energy"] = "DVN"
SCANNER_UNIVERSE["Dexcom"] = "DXCM"
SCANNER_UNIVERSE["Electronic Arts"] = "EA"
SCANNER_UNIVERSE["eBay Inc."] = "EBAY"
SCANNER_UNIVERSE["EchoStar"] = "ECHO"
SCANNER_UNIVERSE["Ecolab"] = "ECL"
SCANNER_UNIVERSE["Consolidated Edison"] = "ED"
SCANNER_UNIVERSE["Equifax"] = "EFX"
SCANNER_UNIVERSE["Everest Group"] = "EG"
SCANNER_UNIVERSE["Edison International"] = "EIX"
SCANNER_UNIVERSE["Estée Lauder Companies (The)"] = "EL"
SCANNER_UNIVERSE["Elevance Health"] = "ELV"
SCANNER_UNIVERSE["Emcor"] = "EME"
SCANNER_UNIVERSE["Emerson Electric"] = "EMR"
SCANNER_UNIVERSE["EOG Resources"] = "EOG"
SCANNER_UNIVERSE["Equinix"] = "EQIX"
SCANNER_UNIVERSE["Equity Residential"] = "EQR"
SCANNER_UNIVERSE["EQT Corporation"] = "EQT"
SCANNER_UNIVERSE["Erie Indemnity"] = "ERIE"
SCANNER_UNIVERSE["Eversource Energy"] = "ES"
SCANNER_UNIVERSE["Essex Property Trust"] = "ESS"
SCANNER_UNIVERSE["Eaton Corporation"] = "ETN"
SCANNER_UNIVERSE["Entergy"] = "ETR"
SCANNER_UNIVERSE["Evergy"] = "EVRG"
SCANNER_UNIVERSE["Edwards Lifesciences"] = "EW"
SCANNER_UNIVERSE["Exelon"] = "EXC"
SCANNER_UNIVERSE["Expand Energy"] = "EXE"
SCANNER_UNIVERSE["Expeditors International"] = "EXPD"
SCANNER_UNIVERSE["Expedia Group"] = "EXPE"
SCANNER_UNIVERSE["Extra Space Storage"] = "EXR"
SCANNER_UNIVERSE["Ford Motor Company"] = "F"
SCANNER_UNIVERSE["Diamondback Energy"] = "FANG"
SCANNER_UNIVERSE["Fastenal"] = "FAST"
SCANNER_UNIVERSE["Freeport-McMoRan"] = "FCX"
SCANNER_UNIVERSE["FactSet"] = "FDS"
SCANNER_UNIVERSE["FedEx"] = "FDX"
SCANNER_UNIVERSE["FedEx Freight"] = "FDXF"
SCANNER_UNIVERSE["FirstEnergy"] = "FE"
SCANNER_UNIVERSE["F5, Inc."] = "FFIV"
SCANNER_UNIVERSE["Fair Isaac"] = "FICO"
SCANNER_UNIVERSE["Fidelity National Information Services"] = "FIS"
SCANNER_UNIVERSE["Fiserv"] = "FISV"
SCANNER_UNIVERSE["Fifth Third Bancorp"] = "FITB"
SCANNER_UNIVERSE["Comfort Systems USA"] = "FIX"
SCANNER_UNIVERSE["Flex Ltd."] = "FLEX"
SCANNER_UNIVERSE["Fox Corporation (Class B)"] = "FOX"
SCANNER_UNIVERSE["Fox Corporation (Class A)"] = "FOXA"
SCANNER_UNIVERSE["Federal Realty Investment Trust"] = "FRT"
SCANNER_UNIVERSE["First Solar"] = "FSLR"
SCANNER_UNIVERSE["Fortinet"] = "FTNT"
SCANNER_UNIVERSE["Fortive"] = "FTV"
SCANNER_UNIVERSE["General Dynamics"] = "GD"
SCANNER_UNIVERSE["GoDaddy"] = "GDDY"
SCANNER_UNIVERSE["GE Aerospace"] = "GE"
SCANNER_UNIVERSE["GE HealthCare"] = "GEHC"
SCANNER_UNIVERSE["Gen Digital"] = "GEN"
SCANNER_UNIVERSE["GE Vernova"] = "GEV"
SCANNER_UNIVERSE["Gilead Sciences"] = "GILD"
SCANNER_UNIVERSE["General Mills"] = "GIS"
SCANNER_UNIVERSE["Globe Life"] = "GL"
SCANNER_UNIVERSE["Corning Inc."] = "GLW"
SCANNER_UNIVERSE["General Motors"] = "GM"
SCANNER_UNIVERSE["Generac"] = "GNRC"
SCANNER_UNIVERSE["Alphabet Inc. (Class C)"] = "GOOG"
SCANNER_UNIVERSE["Alphabet Inc. (Class A)"] = "GOOGL"
SCANNER_UNIVERSE["Genuine Parts Company"] = "GPC"
SCANNER_UNIVERSE["Global Payments"] = "GPN"
SCANNER_UNIVERSE["Garmin"] = "GRMN"
SCANNER_UNIVERSE["Goldman Sachs"] = "GS"
SCANNER_UNIVERSE["W. W. Grainger"] = "GWW"
SCANNER_UNIVERSE["Halliburton"] = "HAL"
SCANNER_UNIVERSE["Hasbro"] = "HAS"
SCANNER_UNIVERSE["Huntington Bancshares"] = "HBAN"
SCANNER_UNIVERSE["HCA Healthcare"] = "HCA"
SCANNER_UNIVERSE["Home Depot (The)"] = "HD"
SCANNER_UNIVERSE["Hartford (The)"] = "HIG"
SCANNER_UNIVERSE["Huntington Ingalls Industries"] = "HII"
SCANNER_UNIVERSE["Hilton Worldwide"] = "HLT"
SCANNER_UNIVERSE["Honeywell Technologies"] = "HON"
SCANNER_UNIVERSE["Honeywell Aerospace"] = "HONA"
SCANNER_UNIVERSE["Robinhood Markets"] = "HOOD"
SCANNER_UNIVERSE["Hewlett Packard Enterprise"] = "HPE"
SCANNER_UNIVERSE["HP Inc."] = "HPQ"
SCANNER_UNIVERSE["Hormel Foods"] = "HRL"
SCANNER_UNIVERSE["Henry Schein"] = "HSIC"
SCANNER_UNIVERSE["Host Hotels & Resorts"] = "HST"
SCANNER_UNIVERSE["Hershey Company (The)"] = "HSY"
SCANNER_UNIVERSE["Hubbell Incorporated"] = "HUBB"
SCANNER_UNIVERSE["Humana"] = "HUM"
SCANNER_UNIVERSE["Howmet Aerospace"] = "HWM"
SCANNER_UNIVERSE["Interactive Brokers"] = "IBKR"
SCANNER_UNIVERSE["IBM"] = "IBM"
SCANNER_UNIVERSE["Intercontinental Exchange"] = "ICE"
SCANNER_UNIVERSE["Idexx Laboratories"] = "IDXX"
SCANNER_UNIVERSE["IDEX Corporation"] = "IEX"
SCANNER_UNIVERSE["International Flavors & Fragrances"] = "IFF"
SCANNER_UNIVERSE["Incyte"] = "INCY"
SCANNER_UNIVERSE["Intel"] = "INTC"
SCANNER_UNIVERSE["Intuit"] = "INTU"
SCANNER_UNIVERSE["Invitation Homes"] = "INVH"
SCANNER_UNIVERSE["International Paper"] = "IP"
SCANNER_UNIVERSE["IQVIA"] = "IQV"
SCANNER_UNIVERSE["Ingersoll Rand"] = "IR"
SCANNER_UNIVERSE["Iron Mountain"] = "IRM"
SCANNER_UNIVERSE["Intuitive Surgical"] = "ISRG"
SCANNER_UNIVERSE["Gartner"] = "IT"
SCANNER_UNIVERSE["Illinois Tool Works"] = "ITW"
SCANNER_UNIVERSE["Invesco"] = "IVZ"
SCANNER_UNIVERSE["Jacobs Solutions"] = "J"
SCANNER_UNIVERSE["J.B. Hunt"] = "JBHT"
SCANNER_UNIVERSE["Jabil"] = "JBL"
SCANNER_UNIVERSE["Johnson Controls"] = "JCI"
SCANNER_UNIVERSE["Jack Henry & Associates"] = "JKHY"
SCANNER_UNIVERSE["Johnson & Johnson"] = "JNJ"
SCANNER_UNIVERSE["JPMorgan Chase"] = "JPM"
SCANNER_UNIVERSE["Keurig Dr Pepper"] = "KDP"
SCANNER_UNIVERSE["KeyCorp"] = "KEY"
SCANNER_UNIVERSE["Keysight Technologies"] = "KEYS"
SCANNER_UNIVERSE["Kraft Heinz"] = "KHC"
SCANNER_UNIVERSE["Kimco Realty"] = "KIM"
SCANNER_UNIVERSE["KKR & Co."] = "KKR"
SCANNER_UNIVERSE["KLA Corporation"] = "KLAC"
SCANNER_UNIVERSE["Kimberly-Clark"] = "KMB"
SCANNER_UNIVERSE["Kinder Morgan"] = "KMI"
SCANNER_UNIVERSE["Coca-Cola Company (The)"] = "KO"
SCANNER_UNIVERSE["Kroger"] = "KR"
SCANNER_UNIVERSE["Kenvue"] = "KVUE"
SCANNER_UNIVERSE["Loews Corporation"] = "L"
SCANNER_UNIVERSE["Leidos"] = "LDOS"
SCANNER_UNIVERSE["Lennar"] = "LEN"
SCANNER_UNIVERSE["Labcorp"] = "LH"
SCANNER_UNIVERSE["L3Harris"] = "LHX"
SCANNER_UNIVERSE["Lennox International"] = "LII"
SCANNER_UNIVERSE["Linde plc"] = "LIN"
SCANNER_UNIVERSE["Lumentum"] = "LITE"
SCANNER_UNIVERSE["Lilly (Eli)"] = "LLY"
SCANNER_UNIVERSE["Lockheed Martin"] = "LMT"
SCANNER_UNIVERSE["Alliant Energy"] = "LNT"
SCANNER_UNIVERSE["Lowe's"] = "LOW"
SCANNER_UNIVERSE["Lam Research"] = "LRCX"
SCANNER_UNIVERSE["Lululemon Athletica"] = "LULU"
SCANNER_UNIVERSE["Southwest Airlines"] = "LUV"
SCANNER_UNIVERSE["Las Vegas Sands"] = "LVS"
SCANNER_UNIVERSE["LyondellBasell"] = "LYB"
SCANNER_UNIVERSE["Live Nation Entertainment"] = "LYV"
SCANNER_UNIVERSE["Mastercard"] = "MA"
SCANNER_UNIVERSE["Mid-America Apartment Communities"] = "MAA"
SCANNER_UNIVERSE["Marriott International"] = "MAR"
SCANNER_UNIVERSE["Masco"] = "MAS"
SCANNER_UNIVERSE["McDonald's"] = "MCD"
SCANNER_UNIVERSE["Microchip Technology"] = "MCHP"
SCANNER_UNIVERSE["McKesson Corporation"] = "MCK"
SCANNER_UNIVERSE["Moody's Corporation"] = "MCO"
SCANNER_UNIVERSE["Mondelez International"] = "MDLZ"
SCANNER_UNIVERSE["Medtronic"] = "MDT"
SCANNER_UNIVERSE["MetLife"] = "MET"
SCANNER_UNIVERSE["Meta Platforms"] = "META"
SCANNER_UNIVERSE["MGM Resorts"] = "MGM"
SCANNER_UNIVERSE["McCormick & Company"] = "MKC"
SCANNER_UNIVERSE["Martin Marietta Materials"] = "MLM"
SCANNER_UNIVERSE["3M"] = "MMM"
SCANNER_UNIVERSE["Monster Beverage"] = "MNST"
SCANNER_UNIVERSE["Altria"] = "MO"
SCANNER_UNIVERSE["Mosaic Company (The)"] = "MOS"
SCANNER_UNIVERSE["Marathon Petroleum"] = "MPC"
SCANNER_UNIVERSE["Monolithic Power Systems"] = "MPWR"
SCANNER_UNIVERSE["Merck & Co."] = "MRK"
SCANNER_UNIVERSE["Moderna"] = "MRNA"
SCANNER_UNIVERSE["Marsh McLennan"] = "MRSH"
SCANNER_UNIVERSE["Marvell Technology"] = "MRVL"
SCANNER_UNIVERSE["Morgan Stanley"] = "MS"
SCANNER_UNIVERSE["MSCI Inc."] = "MSCI"
SCANNER_UNIVERSE["Microsoft"] = "MSFT"
SCANNER_UNIVERSE["Motorola Solutions"] = "MSI"
SCANNER_UNIVERSE["M&T Bank"] = "MTB"
SCANNER_UNIVERSE["Mettler Toledo"] = "MTD"
SCANNER_UNIVERSE["Micron Technology"] = "MU"
SCANNER_UNIVERSE["Norwegian Cruise Line Holdings"] = "NCLH"
SCANNER_UNIVERSE["Nasdaq, Inc."] = "NDAQ"
SCANNER_UNIVERSE["Nordson Corporation"] = "NDSN"
SCANNER_UNIVERSE["NextEra Energy"] = "NEE"
SCANNER_UNIVERSE["Newmont"] = "NEM"
SCANNER_UNIVERSE["Netflix"] = "NFLX"
SCANNER_UNIVERSE["NiSource"] = "NI"
SCANNER_UNIVERSE["Nike, Inc."] = "NKE"
SCANNER_UNIVERSE["Northrop Grumman"] = "NOC"
SCANNER_UNIVERSE["ServiceNow"] = "NOW"
SCANNER_UNIVERSE["NRG Energy"] = "NRG"
SCANNER_UNIVERSE["Norfolk Southern"] = "NSC"
SCANNER_UNIVERSE["NetApp"] = "NTAP"
SCANNER_UNIVERSE["Northern Trust"] = "NTRS"
SCANNER_UNIVERSE["Nucor"] = "NUE"
SCANNER_UNIVERSE["Nvidia"] = "NVDA"
SCANNER_UNIVERSE["NVR, Inc."] = "NVR"
SCANNER_UNIVERSE["News Corp (Class B)"] = "NWS"
SCANNER_UNIVERSE["News Corp (Class A)"] = "NWSA"
SCANNER_UNIVERSE["NXP Semiconductors"] = "NXPI"
SCANNER_UNIVERSE["Realty Income"] = "O"
SCANNER_UNIVERSE["Old Dominion"] = "ODFL"
SCANNER_UNIVERSE["Oneok"] = "OKE"
SCANNER_UNIVERSE["Omnicom Group"] = "OMC"
SCANNER_UNIVERSE["ON Semiconductor"] = "ON"
SCANNER_UNIVERSE["Oracle Corporation"] = "ORCL"
SCANNER_UNIVERSE["O'Reilly Automotive"] = "ORLY"
SCANNER_UNIVERSE["Otis Worldwide"] = "OTIS"
SCANNER_UNIVERSE["Occidental Petroleum"] = "OXY"
SCANNER_UNIVERSE["Palo Alto Networks"] = "PANW"
SCANNER_UNIVERSE["Paychex"] = "PAYX"
SCANNER_UNIVERSE["Paccar"] = "PCAR"
SCANNER_UNIVERSE["PG&E Corporation"] = "PCG"
SCANNER_UNIVERSE["Public Service Enterprise Group"] = "PEG"
SCANNER_UNIVERSE["PepsiCo"] = "PEP"
SCANNER_UNIVERSE["Pfizer"] = "PFE"
SCANNER_UNIVERSE["Principal Financial Group"] = "PFG"
SCANNER_UNIVERSE["Procter & Gamble"] = "PG"
SCANNER_UNIVERSE["Progressive Corporation"] = "PGR"
SCANNER_UNIVERSE["Parker Hannifin"] = "PH"
SCANNER_UNIVERSE["PulteGroup"] = "PHM"
SCANNER_UNIVERSE["Packaging Corporation of America"] = "PKG"
SCANNER_UNIVERSE["Prologis"] = "PLD"
SCANNER_UNIVERSE["Palantir Technologies"] = "PLTR"
SCANNER_UNIVERSE["Philip Morris International"] = "PM"
SCANNER_UNIVERSE["PNC Financial Services"] = "PNC"
SCANNER_UNIVERSE["Pentair"] = "PNR"
SCANNER_UNIVERSE["Pinnacle West Capital"] = "PNW"
SCANNER_UNIVERSE["Insulet Corporation"] = "PODD"
SCANNER_UNIVERSE["PPG Industries"] = "PPG"
SCANNER_UNIVERSE["PPL Corporation"] = "PPL"
SCANNER_UNIVERSE["Prudential Financial"] = "PRU"
SCANNER_UNIVERSE["Public Storage"] = "PSA"
SCANNER_UNIVERSE["Paramount Skydance Corporation"] = "PSKY"
SCANNER_UNIVERSE["Phillips 66"] = "PSX"
SCANNER_UNIVERSE["PTC Inc."] = "PTC"
SCANNER_UNIVERSE["Quanta Services"] = "PWR"
SCANNER_UNIVERSE["PayPal"] = "PYPL"
SCANNER_UNIVERSE["Qualcomm"] = "QCOM"
SCANNER_UNIVERSE["Royal Caribbean Group"] = "RCL"
SCANNER_UNIVERSE["Regency Centers"] = "REG"
SCANNER_UNIVERSE["Regeneron Pharmaceuticals"] = "REGN"
SCANNER_UNIVERSE["Regions Financial Corporation"] = "RF"
SCANNER_UNIVERSE["Raymond James Financial"] = "RJF"
SCANNER_UNIVERSE["Ralph Lauren Corporation"] = "RL"
SCANNER_UNIVERSE["ResMed"] = "RMD"
SCANNER_UNIVERSE["Rockwell Automation"] = "ROK"
SCANNER_UNIVERSE["Rollins, Inc."] = "ROL"
SCANNER_UNIVERSE["Roper Technologies"] = "ROP"
SCANNER_UNIVERSE["Ross Stores"] = "ROST"
SCANNER_UNIVERSE["Republic Services"] = "RSG"
SCANNER_UNIVERSE["RTX Corporation"] = "RTX"
SCANNER_UNIVERSE["Revvity"] = "RVTY"
SCANNER_UNIVERSE["SBA Communications"] = "SBAC"
SCANNER_UNIVERSE["Starbucks"] = "SBUX"
SCANNER_UNIVERSE["Charles Schwab Corporation"] = "SCHW"
SCANNER_UNIVERSE["Sherwin-Williams"] = "SHW"
SCANNER_UNIVERSE["J.M. Smucker Company (The)"] = "SJM"
SCANNER_UNIVERSE["Schlumberger"] = "SLB"
SCANNER_UNIVERSE["Supermicro"] = "SMCI"
SCANNER_UNIVERSE["Snap-on"] = "SNA"
SCANNER_UNIVERSE["Sandisk"] = "SNDK"
SCANNER_UNIVERSE["Synopsys"] = "SNPS"
SCANNER_UNIVERSE["Southern Company"] = "SO"
SCANNER_UNIVERSE["Solventum"] = "SOLV"
SCANNER_UNIVERSE["Simon Property Group"] = "SPG"
SCANNER_UNIVERSE["S&P Global"] = "SPGI"
SCANNER_UNIVERSE["Sempra"] = "SRE"
SCANNER_UNIVERSE["Steris"] = "STE"
SCANNER_UNIVERSE["Steel Dynamics"] = "STLD"
SCANNER_UNIVERSE["State Street Corporation"] = "STT"
SCANNER_UNIVERSE["Seagate Technology"] = "STX"
SCANNER_UNIVERSE["Constellation Brands"] = "STZ"
SCANNER_UNIVERSE["Smurfit Westrock"] = "SW"
SCANNER_UNIVERSE["Stanley Black & Decker"] = "SWK"
SCANNER_UNIVERSE["Skyworks Solutions"] = "SWKS"
SCANNER_UNIVERSE["Synchrony Financial"] = "SYF"
SCANNER_UNIVERSE["Stryker Corporation"] = "SYK"
SCANNER_UNIVERSE["Sysco"] = "SYY"
SCANNER_UNIVERSE["AT&T"] = "T"
SCANNER_UNIVERSE["Molson Coors Beverage Company"] = "TAP"
SCANNER_UNIVERSE["TransDigm Group"] = "TDG"
SCANNER_UNIVERSE["Teledyne Technologies"] = "TDY"
SCANNER_UNIVERSE["Bio-Techne"] = "TECH"
SCANNER_UNIVERSE["TE Connectivity"] = "TEL"
SCANNER_UNIVERSE["Teradyne"] = "TER"
SCANNER_UNIVERSE["Truist Financial"] = "TFC"
SCANNER_UNIVERSE["Target Corporation"] = "TGT"
SCANNER_UNIVERSE["TJX Companies"] = "TJX"
SCANNER_UNIVERSE["TKO Group Holdings"] = "TKO"
SCANNER_UNIVERSE["Thermo Fisher Scientific"] = "TMO"
SCANNER_UNIVERSE["T-Mobile US"] = "TMUS"
SCANNER_UNIVERSE["Texas Pacific Land Corporation"] = "TPL"
SCANNER_UNIVERSE["Tapestry, Inc."] = "TPR"
SCANNER_UNIVERSE["Targa Resources"] = "TRGP"
SCANNER_UNIVERSE["Trimble Inc."] = "TRMB"
SCANNER_UNIVERSE["T. Rowe Price"] = "TROW"
SCANNER_UNIVERSE["Travelers Companies (The)"] = "TRV"
SCANNER_UNIVERSE["Tractor Supply"] = "TSCO"
SCANNER_UNIVERSE["Tesla, Inc."] = "TSLA"
SCANNER_UNIVERSE["Tyson Foods"] = "TSN"
SCANNER_UNIVERSE["Trane Technologies"] = "TT"
SCANNER_UNIVERSE["Trade Desk (The)"] = "TTD"
SCANNER_UNIVERSE["Take-Two Interactive"] = "TTWO"
SCANNER_UNIVERSE["Texas Instruments"] = "TXN"
SCANNER_UNIVERSE["Textron"] = "TXT"
SCANNER_UNIVERSE["Tyler Technologies"] = "TYL"
SCANNER_UNIVERSE["United Airlines Holdings"] = "UAL"
SCANNER_UNIVERSE["Uber"] = "UBER"
SCANNER_UNIVERSE["UDR, Inc."] = "UDR"
SCANNER_UNIVERSE["Universal Health Services"] = "UHS"
SCANNER_UNIVERSE["Ulta Beauty"] = "ULTA"
SCANNER_UNIVERSE["UnitedHealth Group"] = "UNH"
SCANNER_UNIVERSE["Union Pacific Corporation"] = "UNP"
SCANNER_UNIVERSE["United Parcel Service"] = "UPS"
SCANNER_UNIVERSE["United Rentals"] = "URI"
SCANNER_UNIVERSE["U.S. Bancorp"] = "USB"
SCANNER_UNIVERSE["Visa Inc."] = "V"
SCANNER_UNIVERSE["Veeva Systems"] = "VEEV"
SCANNER_UNIVERSE["Vici Properties"] = "VICI"
SCANNER_UNIVERSE["Valero Energy"] = "VLO"
SCANNER_UNIVERSE["Veralto"] = "VLTO"
SCANNER_UNIVERSE["Vulcan Materials Company"] = "VMC"
SCANNER_UNIVERSE["Verisk Analytics"] = "VRSK"
SCANNER_UNIVERSE["Verisign"] = "VRSN"
SCANNER_UNIVERSE["Vertiv"] = "VRT"
SCANNER_UNIVERSE["Vertex Pharmaceuticals"] = "VRTX"
SCANNER_UNIVERSE["Vistra Corp."] = "VST"
SCANNER_UNIVERSE["Ventas"] = "VTR"
SCANNER_UNIVERSE["Viatris"] = "VTRS"
SCANNER_UNIVERSE["Verizon"] = "VZ"
SCANNER_UNIVERSE["Wabtec"] = "WAB"
SCANNER_UNIVERSE["Waters Corporation"] = "WAT"
SCANNER_UNIVERSE["Warner Bros. Discovery"] = "WBD"
SCANNER_UNIVERSE["Workday, Inc."] = "WDAY"
SCANNER_UNIVERSE["Western Digital"] = "WDC"
SCANNER_UNIVERSE["WEC Energy Group"] = "WEC"
SCANNER_UNIVERSE["Welltower"] = "WELL"
SCANNER_UNIVERSE["Wells Fargo"] = "WFC"
SCANNER_UNIVERSE["Waste Management"] = "WM"
SCANNER_UNIVERSE["Williams Companies"] = "WMB"
SCANNER_UNIVERSE["Walmart"] = "WMT"
SCANNER_UNIVERSE["W. R. Berkley Corporation"] = "WRB"
SCANNER_UNIVERSE["Williams-Sonoma, Inc."] = "WSM"
SCANNER_UNIVERSE["West Pharmaceutical Services"] = "WST"
SCANNER_UNIVERSE["Willis Towers Watson"] = "WTW"
SCANNER_UNIVERSE["Weyerhaeuser"] = "WY"
SCANNER_UNIVERSE["Wynn Resorts"] = "WYNN"
SCANNER_UNIVERSE["Xcel Energy"] = "XEL"
SCANNER_UNIVERSE["ExxonMobil"] = "XOM"
SCANNER_UNIVERSE["Xylem Inc."] = "XYL"
SCANNER_UNIVERSE["Block, Inc."] = "XYZ"
SCANNER_UNIVERSE["Yum! Brands"] = "YUM"
SCANNER_UNIVERSE["Zimmer Biomet"] = "ZBH"
SCANNER_UNIVERSE["Zebra Technologies"] = "ZBRA"
SCANNER_UNIVERSE["Zoetis"] = "ZTS"

@st.cache_resource(show_spinner=False)
def get_supabase_client():
    st.session_state["_supabase_last_error"] = None
    if not SUPABASE_PACKAGE_AVAILABLE:
        st.session_state["_supabase_last_error"] = (
            "Das Python-Paket 'supabase' ist nicht installiert (Import ist fehlgeschlagen). "
            "Bitte in requirements.txt prüfen, ob dort 'supabase' steht, und die App neu deployen."
        )
        return None
    try:
        url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
        key = (
            st.secrets.get("SUPABASE_SECRET_KEY")
            or os.getenv("SUPABASE_SECRET_KEY")
            or st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )
    except Exception:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        st.session_state["_supabase_last_error"] = (
            f"SUPABASE_URL {'gefunden' if url else 'FEHLT'}, "
            f"SUPABASE_SECRET_KEY/SUPABASE_SERVICE_ROLE_KEY {'gefunden' if key else 'FEHLT'} "
            "in st.secrets bzw. den Umgebungsvariablen."
        )
        return None
    try:
        return create_client(url, key)
    except Exception as error:
        st.session_state["_supabase_last_error"] = f"create_client() ist fehlgeschlagen: {error!r}"
        return None

def learning_db_ready() -> bool:
    return get_supabase_client() is not None

def _event_key(symbol: str, interval_key: str, timestamp) -> str:
    raw = f"{symbol}|{interval_key}|{LEARNING_FEATURE_VERSION}|{timestamp}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def save_learning_examples(symbol: str, interval_key: str, feature_df: pd.DataFrame, feature_cols: list[str]) -> int:
    client = get_supabase_client()
    if client is None:
        return 0
    usable = feature_df.dropna(subset=feature_cols + ["target"]).copy()
    if usable.empty:
        return 0
    rows = []
    for idx, row in usable.iterrows():
        timestamp = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        features = {name: float(row[name]) for name in feature_cols}
        future_return = row.get("future_return")
        rows.append({
            "symbol": symbol,
            "interval_key": interval_key,
            "feature_version": LEARNING_FEATURE_VERSION,
            "features": features,
            "target": int(row["target"]),
            "future_return": float(future_return) if pd.notna(future_return) else None,
            "label_time": timestamp,
            "source": "paper",
            "event_key": _event_key(symbol, interval_key, timestamp),
        })
    inserted = 0
    for start in range(0, len(rows), 500):
        batch = rows[start:start + 500]
        try:
            client.table("learning_examples").upsert(batch, on_conflict="event_key").execute()
            # Supabase may return no row body for a successful upsert. Count
            # the successfully synchronized rows instead of showing a false 0.
            inserted += len(batch)
        except Exception as exc:
            st.session_state.learning_sync_error = str(exc)
    return inserted

def collect_shared_learning(symbol: str, interval_key: str, df: pd.DataFrame) -> None:
    """Add anonymized market features whenever somebody uses the app."""
    if not learning_db_ready() or df.empty or len(df) < 250:
        return
    try:
        feature_df, feature_cols = build_ml_features(df)
        save_learning_examples(symbol, interval_key, feature_df, feature_cols)
        status = get_learning_status()
        state = get_supabase_client().table("learning_state").select("value").eq("key", "global").maybe_single().execute().data or {}
        examples_at_last_training = int((state.get("value") or {}).get("examples_seen") or 0)
        if status["examples"] >= TRAINING_MIN_SAMPLES and status["examples"] - examples_at_last_training >= AUTO_TRAIN_MIN_NEW_EXAMPLES:
            train_and_maybe_promote_shared_model()
    except Exception:
        # Analysis must stay available if the optional learning service is offline.
        return

def load_learning_examples() -> pd.DataFrame:
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()
    rows = []
    try:
        offset = 0
        while True:
            result = (
                client.table("learning_examples")
                .select("features,target,symbol,interval_key,label_time,created_at")
                .eq("feature_version", LEARNING_FEATURE_VERSION)
                .order("created_at", desc=False)
                .range(offset, offset + 999)
                .execute()
            )
            batch = result.data or []
            rows.extend(batch)
            if len(batch) < 1000:
                break
            offset += 1000
            if offset >= 50000:
                break
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    records = []
    for item in rows:
        record = dict(item.get("features") or {})
        record["target"] = int(item["target"])
        record["symbol"] = item.get("symbol")
        record["interval_key"] = item.get("interval_key")
        # created_at says when a user uploaded a candle, not when the candle
        # existed. Training by it would leak newer market data into earlier
        # validation folds when users sync assets at different times.
        record["label_time"] = item.get("label_time") or item.get("created_at")
        records.append(record)
    data = pd.DataFrame(records)
    data["label_time"] = pd.to_datetime(data["label_time"], utc=True, errors="coerce")
    return data.sort_values("label_time", kind="stable").reset_index(drop=True)

def get_learning_status() -> dict:
    client = get_supabase_client()
    if client is None:
        return {"ready": False, "examples": 0, "active_model": None, "last_training": None}
    try:
        state = client.table("learning_state").select("value").eq("key", "global").maybe_single().execute().data
        value = (state or {}).get("value") or {}
        count = client.table("learning_examples").select("id", count="exact").eq("feature_version", LEARNING_FEATURE_VERSION).execute().count or 0
        return {
            "ready": True,
            "examples": int(count),
            "active_model": value.get("active_model_id"),
            "last_training": value.get("last_training_at"),
        }
    except Exception:
        return {"ready": True, "examples": 0, "active_model": None, "last_training": None}

def _set_learning_state(**updates):
    client = get_supabase_client()
    if client is None:
        return
    try:
        current = client.table("learning_state").select("value").eq("key", "global").maybe_single().execute().data
        value = dict((current or {}).get("value") or {})
        value.update(updates)
        client.table("learning_state").upsert({"key": "global", "value": value}).execute()
    except Exception:
        pass

def _latest_promoted_model():
    client = get_supabase_client()
    if client is None:
        return None
    try:
        result = (
            client.table("model_versions")
            .select("*")
            .eq("promoted", True)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return (result.data or [None])[0]
    except Exception:
        return None

def train_and_maybe_promote_shared_model() -> dict:
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn fehlt."}
    client = get_supabase_client()
    if client is None:
        return {"error": "Supabase ist noch nicht verbunden. Hinterlege SUPABASE_URL und SUPABASE_SECRET_KEY in den Streamlit-Secrets."}

    data = load_learning_examples()
    feature_cols = ["rsi", "macd_hist", "bb_pos", "ema_gap", "ret_1", "ret_5", "ret_10", "vol_ratio", "body_ratio"]
    usable = data.dropna(subset=feature_cols + ["target"]).copy() if not data.empty else pd.DataFrame()
    if len(usable) < TRAINING_MIN_SAMPLES:
        return {"error": f"Noch zu wenig gemeinsame Trainingsdaten: {len(usable)}/{TRAINING_MIN_SAMPLES}."}

    run = client.table("training_runs").insert({"status": "running", "sample_count": int(len(usable))}).execute()
    run_id = (run.data or [{}])[0].get("id")
    try:
        n_folds = 5
        fold_size = len(usable) // (n_folds + 1)
        fold_accuracies = []
        fold_details = []
        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            test_end = fold_size * (fold + 2)
            train_slice = usable.iloc[:train_end]
            test_slice = usable.iloc[train_end:test_end]
            if len(train_slice) < 50 or len(test_slice) < 10:
                continue
            model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train_slice[feature_cols], train_slice["target"])
            accuracy = float((model.predict(test_slice[feature_cols]) == test_slice["target"].values).mean())
            fold_accuracies.append(accuracy)
            fold_details.append({"fold": fold + 1, "train_size": len(train_slice), "test_size": len(test_slice), "accuracy": round(accuracy * 100, 1)})
        if not fold_accuracies:
            raise RuntimeError("Keine gültigen Walk-Forward-Folds möglich.")

        new_accuracy = float(np.mean(fold_accuracies))
        positive_rate = float(usable["target"].mean())
        # Accuracy is only meaningful if it beats the trivial majority-class
        # predictor. For an imbalanced data set, "always up" can look good.
        baseline = max(positive_rate, 1 - positive_rate)
        final_model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
        final_model.fit(usable[feature_cols], usable["target"])
        metrics = {
            "mean_accuracy": round(new_accuracy * 100, 2),
            "baseline_accuracy": round(baseline * 100, 2),
            "up_rate": round(positive_rate * 100, 2),
            "fold_details": fold_details,
            "sample_size": int(len(usable)),
            "feature_importances": dict(zip(feature_cols, final_model.feature_importances_.round(4))),
        }
        previous = _latest_promoted_model()
        previous_accuracy = None
        if previous:
            previous_accuracy = ((previous.get("validation_metrics") or {}).get("mean_accuracy"))

        # A candidate must beat a naive baseline and must not degrade the
        # active model. This keeps noise from becoming the production model.
        beats_baseline = new_accuracy >= baseline
        promote = beats_baseline and (previous is None or previous_accuracy is None or new_accuracy * 100 >= float(previous_accuracy))
        model_row = {
            "training_run_id": run_id,
            "algorithm": MODEL_ALGORITHM,
            "feature_version": LEARNING_FEATURE_VERSION,
            "sample_count": int(len(usable)),
            "validation_metrics": metrics,
            # Do not store pickle payloads in a database: loading an altered
            # pickle can execute arbitrary code. The app only needs metrics
            # and retrains from verified examples on demand.
            "model_artifact_base64": None,
            "promoted": bool(promote),
            "rejection_reason": None if promote else (
                f"Validation {new_accuracy*100:.2f}% did not beat baseline {baseline*100:.2f}%"
                if not beats_baseline else f"Validation {new_accuracy*100:.2f}% < active {float(previous_accuracy):.2f}%"
            ),
        }
        if promote and previous:
            client.table("model_versions").update({"promoted": False}).eq("promoted", True).execute()
        model_insert = client.table("model_versions").insert(model_row).execute()
        model_id = (model_insert.data or [{}])[0].get("id")
        now = datetime.now(ZoneInfo("UTC")).isoformat()
        client.table("training_runs").update({"status": "completed" if promote else "rejected", "finished_at": now, "metrics": metrics}).eq("id", run_id).execute()
        _set_learning_state(
            active_model_id=model_id if promote else (previous or {}).get("id"),
            examples_seen=int(len(usable)), last_training_at=now,
            feature_version=LEARNING_FEATURE_VERSION,
        )
        return {"ok": True, "promoted": promote, "metrics": metrics, "previous_accuracy": previous_accuracy, "model_id": model_id}
    except Exception as exc:
        client.table("training_runs").update({"status": "failed", "finished_at": datetime.now(ZoneInfo("UTC")).isoformat(), "error_message": str(exc)}).eq("id", run_id).execute()
        return {"error": f"Training fehlgeschlagen: {exc}"}

# ------------------------------------------------------------
# Seiten-Konfiguration (Handy-optimiert)
# ------------------------------------------------------------
_icon_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
st.set_page_config(
    page_title="Pattern Analyzer",
    page_icon=_icon_full_path if os.path.exists(_icon_full_path) else "📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.6em; padding-bottom: 3em; max-width: 600px; }

    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    }
    body, .stApp { background-color: #171c28 !important; color: #e0e5ef; }

    .app-header { margin-bottom: 1.6em; display: flex; align-items: center; gap: 0.6em; }
    .app-brand { text-align: center; margin: 0 auto 1.7em; }
    .app-title {
        font-size: 1.55em;
        font-weight: 700;
        letter-spacing: 0;
        line-height: 1.15;
        color: #ffffff;
        margin-bottom: 0.15em;
    }
    .app-subtitle { color: #aeb8c9; font-size: 0.9em; line-height: 1.35; }

    .section-label {
        font-size: 0.74em;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #aeb8c9;
        margin: 1.5em 0 0.5em 2px;
    }
    .pattern-book-heading {
        display: flex;
        align-items: center;
        gap: 0.65em;
        margin: 1.25em 0 0.25em;
    }
    .pattern-book-icon { font-size: 1.65em; line-height: 1; }
    .pattern-book-heading-title { color: #ffffff; font-size: 1.08em; font-weight: 750; line-height: 1.25; }
    .pattern-book-heading-subtitle { color: #aeb8c9; font-size: 0.76em; line-height: 1.35; margin-top: 0.18em; }

    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] [role="combobox"] {
        border-radius: 10px !important;
        border: 1px solid #3b4354 !important;
        background: #252b39 !important;
        background-color: #252b39 !important;
        color: #e0e5ef !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] [role="combobox"] {
        background: #252b39 !important;
        background-color: #252b39 !important;
        color: #e0e5ef !important;
    }
    div[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"],
    div[data-testid="stSelectbox"] div[role="group"] {
        background: #252b39 !important;
        background-color: #252b39 !important;
        border: 1px solid #4a566c !important;
        border-radius: 10px !important;
    }
    div[data-testid="stSelectbox"] input[role="combobox"] {
        background: transparent !important;
        color: #e0e5ef !important;
        -webkit-text-fill-color: #e0e5ef !important;
    }
    div[data-baseweb="select"] * {
        color: #e0e5ef !important;
    }
    .stTextInput input {
        border-radius: 10px !important;
        border: 1px solid #3b4354 !important;
        background-color: #252b39 !important;
        color: #e0e5ef !important;
    }
    .stTextInput > div > div,
    div[data-baseweb="input"],
    .stNumberInput input,
    .stNumberInput > div > div {
        background-color: #252b39 !important;
        border: 1px solid #4a566c !important;
        color: #e0e5ef !important;
    }
    .stTextInput input::placeholder {
        color: #c8d1df !important;
        opacity: 1 !important;
    }
    .stTextInput label,
    div[data-testid="stExpander"] label {
        color: #dce4f0 !important;
    }
    div[data-testid="stExpander"] {
        border-color: #4a566c !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: #252b39 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #222938 !important;
        padding: 4px;
        border-radius: 10px;
        border: 1px solid #3b4354 !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px !important;
        color: #aeb8c9 !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        background-color: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #35405a !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs button[data-baseweb="tab"]:focus { outline: none !important; box-shadow: none !important; }
    button:focus, button:focus-visible { outline: none !important; box-shadow: none !important; }

    .stButton button {
        width: 100%;
        height: 3.3em;
        font-size: 1.02em;
        font-weight: 650;
        border-radius: 10px;
        border: 1px solid #3b4354;
        background: #252b39;
        color: #ffffff;
        margin-top: 0.3em;
        transition: opacity 0.15s ease;
    }
    .stButton button:hover,
    .stButton button:focus,
    .stButton button:active {
        background: #252b39 !important;
        border-color: #5a6880 !important;
        color: #ffffff !important;
    }
    section[data-testid="stFileUploader"] button {
        background: #252b39 !important;
        border: 1px solid #3b4354 !important;
        color: #e0e5ef !important;
    }
    .stButton button:active { opacity: 0.7; }
    .stButton button:disabled { opacity: 0.35; }

    .cta-btn .stButton button {
        background: #2962ff;
        border: none;
        color: #ffffff;
        font-weight: 700;
    }

    .result-card {
        border-radius: 14px;
        padding: 1.4em 1.3em;
        margin-top: 1.2em;
        background: #252b39;
        border: 1px solid #3b4354;
    }
    .pattern-name { font-size: 1em; font-weight: 650; color: #ffffff; margin-bottom: 0.2em; }
    .pattern-meta { font-size: 0.78em; color: #aeb8c9; margin-bottom: 1.1em; }

    .prob-box {
        padding: 0.7em;
        border-radius: 10px;
        text-align: center;
        font-size: 1.7em;
        font-weight: 750;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
    }
    .neutral-box { background-color: #30384a; color: #b8c2d3; font-size: 1.15em; padding: 0.55em; }

    .chart-card {
        border-radius: 14px;
        padding: 0.9em 0.6em 0.3em 0.6em;
        margin-top: 0.9em;
        background: #252b39;
        border: 1px solid #3b4354;
    }
    .green-box { background-color: rgba(61,214,176,0.18); color: #3dd6b0; }
    .red-box   { background-color: rgba(255,107,107,0.18); color: #ff6b6b; }

    .mini-card {
        border-radius: 12px;
        padding: 0.9em 1.1em;
        margin-top: 0.7em;
        background: #252b39;
        border: 1px solid #3b4354;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .mini-ticker { font-weight: 650; font-size: 0.98em; color: #ffffff; }
    .mini-pattern { font-size: 0.78em; color: #aeb8c9; }
    .mini-prob {
        font-size: 1.2em;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        border-radius: 999px;
        padding: 0.3em 0.75em;
    }

    .ai-card {
        border-radius: 12px;
        padding: 1.05em 1.2em;
        margin-top: 0.8em;
        background: #252b39;
        border: 1px solid #3b4354;
        font-size: 0.9em;
        line-height: 1.55em;
        color: #e0e5ef;
    }
    .ai-label { font-size: 0.72em; font-weight: 700; color: #2962ff; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5em; }

    .info-card {
        border-radius: 12px;
        padding: 1em 1.15em;
        margin: 0.8em 0;
        background: #252b39;
        border: 1px solid #3b4354;
        font-size: 0.84em;
        line-height: 1.5em;
        color: #c5ccda;
    }

    .trade-card {
        border-radius: 14px;
        padding: 1.15em 1.2em;
        margin-top: 0.9em;
        background: #252b39;
        border: 1px solid #3b4354;
    }
    .trade-title { font-size: 1em; font-weight: 700; color: #ffffff; margin-bottom: 0.25em; }
    .trade-subtitle { font-size: 0.78em; color: #aeb8c9; margin-bottom: 0.9em; }
    .trade-signal { font-size: 1.35em; font-weight: 750; margin-bottom: 0.7em; }
    .trade-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.55em; }
    .trade-metric { background: #30384a; border-radius: 9px; padding: 0.65em 0.75em; }
    .trade-metric-label { display: block; font-size: 0.72em; color: #b8c2d3; }
    .trade-metric-value { display: block; font-size: 0.98em; font-weight: 650; color: #ffffff; margin-top: 0.15em; }
    .trade-note { font-size: 0.78em; line-height: 1.45em; color: #c5ccda; margin-top: 0.85em; }

    .pattern-book-card {
        border-radius: 12px;
        padding: 0.85em 0.9em;
        margin: 0.55em 0;
        background: #252b39;
        border: 1px solid #3b4354;
    }
    .pattern-book-title { font-size: 0.95em; font-weight: 700; color: #ffffff; }
    .pattern-book-direction { font-size: 0.72em; font-weight: 700; color: #26a69a; text-transform: uppercase; }
    .pattern-book-direction.bearish { color: #ef5350; }
    .pattern-book-direction.neutral { color: #ffd60a; }
    .pattern-book-text { font-size: 0.8em; line-height: 1.45em; color: #c5ccda; margin-top: 0.5em; }
    .pattern-visual { display: flex; align-items: center; justify-content: center; gap: 0.65em; height: 66px; margin: 0.25em 0 0.45em; background: #171a23; border-radius: 8px; }
    .candle { position: relative; width: 18px; height: 54px; }
    .candle-wick { position: absolute; left: 8px; top: 2px; width: 2px; height: 50px; background: #c5ccda; border-radius: 2px; }
    .candle-body { position: absolute; left: 3px; top: 14px; width: 12px; height: 27px; border-radius: 2px; background: #26a69a; border: 1px solid #26a69a; }
    .candle.bear .candle-body { background: #ef5350; border-color: #ef5350; }
    .candle.small .candle-body { top: 24px; height: 11px; }
    .candle.long .candle-body { top: 7px; height: 41px; }
    .candle.doji .candle-body { top: 27px; height: 3px; background: #ffd60a; border-color: #ffd60a; }
    .candle.dragonfly .candle-body { top: 8px; height: 3px; background: #ffd60a; border-color: #ffd60a; }
    .candle.gravestone .candle-body { top: 43px; height: 3px; background: #ffd60a; border-color: #ffd60a; }

    .watch-chip {
        display: inline-block;
        background: #252b39;
        border: 1px solid #3b4354;
        border-radius: 999px;
        padding: 0.4em 0.95em;
        margin: 0.2em 0.3em 0.2em 0;
        font-size: 0.85em;
        color: #ffffff;
        font-variant-numeric: tabular-nums;
    }

    .disclaimer { font-size: 0.74em; color: #8996aa; margin-top: 1.6em; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-brand">'
    '<div class="app-title">Pattern Analyzer</div>'
    '<div class="app-subtitle">Candlestick- &amp; Marktanalyse</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Gemeinsame Konfiguration
# ------------------------------------------------------------
ASSETS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Nvidia (NVDA)": "NVDA",
}

INTERVAL_CONFIG = {
    "1m":  {"yf_interval": "1m",  "period": "7d",   "resample": None},
    "5m":  {"yf_interval": "5m",  "period": "60d",  "resample": None},
    "15m": {"yf_interval": "15m", "period": "60d",  "resample": None},
    "30m": {"yf_interval": "30m", "period": "60d",  "resample": None},
    "1h":  {"yf_interval": "1h",  "period": "730d", "resample": None},
    "4h":  {"yf_interval": "1h",  "period": "730d", "resample": "4h"},
    "1d":  {"yf_interval": "1d",  "period": "5y",   "resample": None},
    "1wk": {"yf_interval": "1wk", "period": "10y",  "resample": None},
    "1mo": {"yf_interval": "1mo", "period": "max",  "resample": None},
}

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = ""

if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}

if "single_result" not in st.session_state:
    st.session_state.single_result = None

if "single_ai_text" not in st.session_state:
    st.session_state.single_ai_text = None

if "single_ai_key" not in st.session_state:
    st.session_state.single_ai_key = None

if "ibkr_feed" not in st.session_state:
    st.session_state.ibkr_feed = None

if "paper_bot_result" not in st.session_state:
    st.session_state.paper_bot_result = None

if "paper_bot_best_params" not in st.session_state:
    st.session_state.paper_bot_best_params = None

if "live_paper_account" not in st.session_state:
    st.session_state.live_paper_account = None

if "ml_result" not in st.session_state:
    st.session_state.ml_result = None

# ------------------------------------------------------------
# API-Key-Eingabe
# ------------------------------------------------------------
has_any_ai_key = any((st.session_state.gemini_api_key, st.session_state.openai_api_key, st.session_state.anthropic_api_key))
key_label = "KI-Anbieter und API-Keys" if not has_any_ai_key else "KI-Anbieter und API-Keys (gesetzt)"
with st.expander(key_label, expanded=not has_any_ai_key):
    ai_provider = st.selectbox(
        "KI-Anbieter",
        ["Keiner", "Gemini", "OpenAI", "Claude"],
        key="ai_provider",
        help="Nur der ausgewählte Anbieter wird bei einem Klick kontaktiert.",
    )
    if ai_provider == "Gemini":
        st.session_state.gemini_api_key = st.text_input(
            "Gemini API Key", type="password", value=st.session_state.gemini_api_key,
            placeholder="Gemini-Key hier einfügen",
            key="main_api_key",
        )
    elif ai_provider == "OpenAI":
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key", type="password", value=st.session_state.openai_api_key,
            placeholder="OpenAI-Key hier einfügen",
            key="openai_api_key_input",
        )
    elif ai_provider == "Claude":
        st.session_state.anthropic_api_key = st.text_input(
            "Claude API Key", type="password", value=st.session_state.anthropic_api_key,
            placeholder="Claude-Key hier einfügen",
            key="anthropic_api_key_input",
        )
    else:
        st.caption("Wähle zuerst einen KI-Anbieter aus.")
    st.caption("Bleibt nur für diese Sitzung gespeichert (Session State), nicht dauerhaft.")

with st.expander("Marktdatenquelle", expanded=False):
    data_source = st.selectbox(
        "Kursquelle",
        ["Yahoo Finance", "Interactive Brokers Paper-Feed"],
        key="data_source",
        help="Der IBKR-Feed liest nur Paper-Marktdaten und platziert keine Orders.",
    )
    if data_source == "Interactive Brokers Paper-Feed":
        ibkr_host = st.text_input("IBKR Host", value="127.0.0.1", key="ibkr_host")
        ibkr_port = st.number_input("IBKR Paper-Port", min_value=1, value=7497, step=1, key="ibkr_port")
        if IBKRFeed is None:
            st.warning("Für den IBKR-Feed fehlt noch 'ib_insync'. Installiere es mit: python -m pip install ib_insync")
        elif st.button("Paper-Feed verbinden", key="ibkr_connect"):
            try:
                st.session_state.ibkr_feed = IBKRFeed(ibkr_host, int(ibkr_port))
                st.session_state.ibkr_feed.connect()
                st.success("IBKR-Paper-Feed verbunden. Es werden keine Orders platziert.")
            except Exception as error:
                st.session_state.ibkr_feed = None
                st.error(f"IBKR-Verbindung fehlgeschlagen: {error}")

# ------------------------------------------------------------
# Daten laden
# ------------------------------------------------------------
def load_data(ticker: str, interval_key: str, source: str = "Yahoo Finance") -> pd.DataFrame:
    if source == "Interactive Brokers Paper-Feed":
        feed = st.session_state.get("ibkr_feed")
        if feed is None or not feed.connected:
            raise RuntimeError("IBKR-Paper-Feed ist nicht verbunden. TWS/IB Gateway starten und Paper-Feed verbinden.")
        return feed.fetch_bars(ticker, interval_key)

    cfg = INTERVAL_CONFIG[interval_key]
    df = yf.download(ticker, period=cfg["period"], interval=cfg["yf_interval"], progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if cfg["resample"]:
        df = df.resample(cfg["resample"]).agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()

    df = df.dropna()
    df = df.tail(1000)
    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    return df

# ------------------------------------------------------------
# Candlestick-Muster-Erkennung
# ------------------------------------------------------------
def _body(row):
    return abs(row["Close"] - row["Open"])

def _range(row):
    return row["High"] - row["Low"]

def is_hammer(df, i) -> bool:
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    return (lower_wick > 2 * body) and (upper_wick < body) and (body / range_ < 0.35)

def is_shooting_star(df, i) -> bool:
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    return (upper_wick > 2 * body) and (lower_wick < body) and (body / range_ < 0.35)

def is_doji(df, i) -> bool:
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    return body / range_ < 0.08

def is_bullish_engulfing(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    return (
        prev["Close"] < prev["Open"] and row["Close"] > row["Open"]
        and row["Open"] <= prev["Close"] and row["Close"] >= prev["Open"]
    )

def is_bearish_engulfing(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    return (
        prev["Close"] > prev["Open"] and row["Close"] < row["Open"]
        and row["Open"] >= prev["Close"] and row["Close"] <= prev["Open"]
    )

def is_bullish_harami(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    return (
        prev["Close"] < prev["Open"] and row["Close"] > row["Open"]
        and _body(prev) > _body(row) * 1.5
        and row["Open"] >= prev["Close"] and row["Close"] <= prev["Open"]
    )

def is_bearish_harami(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    return (
        prev["Close"] > prev["Open"] and row["Close"] < row["Open"]
        and _body(prev) > _body(row) * 1.5
        and row["Open"] <= prev["Close"] and row["Close"] >= prev["Open"]
    )

def is_piercing_line(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    midpoint = (prev["Open"] + prev["Close"]) / 2
    return (
        prev["Close"] < prev["Open"] and row["Close"] > row["Open"]
        and row["Open"] <= prev["Close"] and row["Close"] > midpoint
        and row["Close"] < prev["Open"]
    )

def is_dark_cloud_cover(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    midpoint = (prev["Open"] + prev["Close"]) / 2
    return (
        prev["Close"] > prev["Open"] and row["Close"] < row["Open"]
        and row["Open"] >= prev["Close"] and row["Close"] < midpoint
        and row["Close"] > prev["Open"]
    )

def is_hanging_man(df, i) -> bool:
    if i < 3:
        return False
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    prior_uptrend = df.iloc[i - 1]["Close"] > df.iloc[i - 3]["Close"]
    return prior_uptrend and lower_wick > 2 * body and upper_wick < body and body / range_ < 0.35

def is_inverted_hammer(df, i) -> bool:
    if i < 3:
        return False
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    prior_downtrend = df.iloc[i - 1]["Close"] < df.iloc[i - 3]["Close"]
    return prior_downtrend and upper_wick > 2 * body and lower_wick < body and body / range_ < 0.35

def is_tweezer_bottom(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    tolerance = max(_range(prev), _range(row), 1e-9) * 0.1
    return (
        prev["Close"] < prev["Open"] and row["Close"] > row["Open"]
        and abs(prev["Low"] - row["Low"]) <= tolerance
    )

def is_tweezer_top(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    tolerance = max(_range(prev), _range(row), 1e-9) * 0.1
    return (
        prev["Close"] > prev["Open"] and row["Close"] < row["Open"]
        and abs(prev["High"] - row["High"]) <= tolerance
    )

def is_morning_star(df, i) -> bool:
    if i < 2:
        return False
    a, b, c = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    a_bear = a["Close"] < a["Open"] and _body(a) / max(_range(a), 1e-9) > 0.4
    b_small = _body(b) / max(_range(b), 1e-9) < 0.35
    c_bull = c["Close"] > c["Open"] and c["Close"] > (a["Open"] + a["Close"]) / 2
    return bool(a_bear and b_small and c_bull)

def is_evening_star(df, i) -> bool:
    if i < 2:
        return False
    a, b, c = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    a_bull = a["Close"] > a["Open"] and _body(a) / max(_range(a), 1e-9) > 0.4
    b_small = _body(b) / max(_range(b), 1e-9) < 0.35
    c_bear = c["Close"] < c["Open"] and c["Close"] < (a["Open"] + a["Close"]) / 2
    return bool(a_bull and b_small and c_bear)

def is_three_white_soldiers(df, i) -> bool:
    if i < 2:
        return False
    window = df.iloc[i - 2:i + 1]
    closes_rising = all(window["Close"].iloc[k] > window["Close"].iloc[k - 1] for k in range(1, 3))
    return bool((window["Close"] > window["Open"]).all() and closes_rising)

def is_three_black_crows(df, i) -> bool:
    if i < 2:
        return False
    window = df.iloc[i - 2:i + 1]
    closes_falling = all(window["Close"].iloc[k] < window["Close"].iloc[k - 1] for k in range(1, 3))
    return bool((window["Close"] < window["Open"]).all() and closes_falling)

def is_bullish_marubozu(df, i) -> bool:
    row = df.iloc[i]
    range_ = _range(row)
    if range_ == 0:
        return False
    return row["Close"] > row["Open"] and _body(row) / range_ > 0.9

def is_bearish_marubozu(df, i) -> bool:
    row = df.iloc[i]
    range_ = _range(row)
    if range_ == 0:
        return False
    return row["Close"] < row["Open"] and _body(row) / range_ > 0.9

def is_spinning_top(df, i) -> bool:
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    return 0.08 <= body / range_ <= 0.3 and upper_wick > body and lower_wick > body

def is_dragonfly_doji(df, i) -> bool:
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    return body / range_ < 0.08 and lower_wick > 2 * upper_wick

def is_gravestone_doji(df, i) -> bool:
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    return body / range_ < 0.08 and upper_wick > 2 * lower_wick

def is_bullish_belt_hold(df, i) -> bool:
    row = df.iloc[i]
    range_ = _range(row)
    if range_ == 0:
        return False
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    return row["Close"] > row["Open"] and _body(row) / range_ > 0.7 and lower_wick / range_ < 0.1

def is_bearish_belt_hold(df, i) -> bool:
    row = df.iloc[i]
    range_ = _range(row)
    if range_ == 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    return row["Close"] < row["Open"] and _body(row) / range_ > 0.7 and upper_wick / range_ < 0.1

def is_three_inside_up(df, i) -> bool:
    if i < 2:
        return False
    a, b, c = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    return (
        a["Close"] < a["Open"] and b["Close"] > b["Open"]
        and _body(b) < _body(a) * 0.6
        and b["Open"] >= a["Close"] and b["Close"] <= a["Open"]
        and c["Close"] > a["Open"]
    )

def is_three_inside_down(df, i) -> bool:
    if i < 2:
        return False
    a, b, c = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    return (
        a["Close"] > a["Open"] and b["Close"] < b["Open"]
        and _body(b) < _body(a) * 0.6
        and b["Open"] <= a["Close"] and b["Close"] >= a["Open"]
        and c["Close"] < a["Open"]
    )

def is_rising_three_methods(df, i) -> bool:
    if i < 4:
        return False
    window = df.iloc[i - 4:i + 1]
    first, last = window.iloc[0], window.iloc[4]
    middle = window.iloc[1:4]
    return (
        first["Close"] > first["Open"] and last["Close"] > last["Open"]
        and (middle["Close"] < middle["Open"]).all()
        and (middle["High"] < first["High"]).all()
        and (middle["Low"] > first["Low"]).all()
        and last["Close"] > first["High"]
    )

def is_falling_three_methods(df, i) -> bool:
    if i < 4:
        return False
    window = df.iloc[i - 4:i + 1]
    first, last = window.iloc[0], window.iloc[4]
    middle = window.iloc[1:4]
    return (
        first["Close"] < first["Open"] and last["Close"] < last["Open"]
        and (middle["Close"] > middle["Open"]).all()
        and (middle["High"] < first["High"]).all()
        and (middle["Low"] > first["Low"]).all()
        and last["Close"] < first["Low"]
    )

def is_bullish_three_line_strike(df, i) -> bool:
    if i < 3:
        return False
    window = df.iloc[i - 3:i + 1]
    first_three = window.iloc[:3]
    last = window.iloc[3]
    return (
        (first_three["Close"] < first_three["Open"]).all()
        and first_three["Close"].iloc[1] < first_three["Close"].iloc[0]
        and first_three["Close"].iloc[2] < first_three["Close"].iloc[1]
        and last["Close"] > last["Open"]
        and last["Open"] <= first_three["Close"].iloc[2]
        and last["Close"] >= first_three["Open"].iloc[0]
    )

def is_bearish_three_line_strike(df, i) -> bool:
    if i < 3:
        return False
    window = df.iloc[i - 3:i + 1]
    first_three = window.iloc[:3]
    last = window.iloc[3]
    return (
        (first_three["Close"] > first_three["Open"]).all()
        and first_three["Close"].iloc[1] > first_three["Close"].iloc[0]
        and first_three["Close"].iloc[2] > first_three["Close"].iloc[1]
        and last["Close"] < last["Open"]
        and last["Open"] >= first_three["Close"].iloc[2]
        and last["Close"] <= first_three["Open"].iloc[0]
    )

def is_bullish_kicker(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    return (
        prev["Close"] < prev["Open"] and row["Close"] > row["Open"]
        and _body(row) / max(_range(row), 1e-9) > 0.6
        and row["Open"] > prev["Open"]
    )

def is_bearish_kicker(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    return (
        prev["Close"] > prev["Open"] and row["Close"] < row["Open"]
        and _body(row) / max(_range(row), 1e-9) > 0.6
        and row["Open"] < prev["Open"]
    )

PATTERNS = {
    "Bullish Engulfing (bullisch)": is_bullish_engulfing,
    "Bearish Engulfing (bärisch)": is_bearish_engulfing,
    "Bullish Harami (bullisch)": is_bullish_harami,
    "Bearish Harami (bärisch)": is_bearish_harami,
    "Piercing Line (bullisch)": is_piercing_line,
    "Dark Cloud Cover (bärisch)": is_dark_cloud_cover,
    "Morning Star (bullisch)": is_morning_star,
    "Evening Star (bärisch)": is_evening_star,
    "3 weiße Soldaten (bullisch)": is_three_white_soldiers,
    "3 schwarze Krähen (bärisch)": is_three_black_crows,
    "Hammer (bullisch)": is_hammer,
    "Hanging Man (bärisch)": is_hanging_man,
    "Inverted Hammer (bullisch)": is_inverted_hammer,
    "Shooting Star (bärisch)": is_shooting_star,
    "Tweezer Bottom (bullisch)": is_tweezer_bottom,
    "Tweezer Top (bärisch)": is_tweezer_top,
    "Doji (Unentschlossenheit)": is_doji,
    "Bullish Marubozu (bullisch)": is_bullish_marubozu,
    "Bearish Marubozu (bärisch)": is_bearish_marubozu,
    "Spinning Top (Unentschlossenheit)": is_spinning_top,
    "Dragonfly Doji (bullisch)": is_dragonfly_doji,
    "Gravestone Doji (bärisch)": is_gravestone_doji,
    "Bullish Belt Hold (bullisch)": is_bullish_belt_hold,
    "Bearish Belt Hold (bärisch)": is_bearish_belt_hold,
    "Three Inside Up (bullisch)": is_three_inside_up,
    "Three Inside Down (bärisch)": is_three_inside_down,
    "Rising Three Methods (bullisch)": is_rising_three_methods,
    "Falling Three Methods (bärisch)": is_falling_three_methods,
    "Bullish Three Line Strike (bullisch)": is_bullish_three_line_strike,
    "Bearish Three Line Strike (bärisch)": is_bearish_three_line_strike,
    "Bullish Kicker (bullisch)": is_bullish_kicker,
    "Bearish Kicker (bärisch)": is_bearish_kicker,
}

PATTERN_EXPLAIN = {
    "Bullish Engulfing (bullisch)": "Eine grüne Kerze schluckt komplett den Körper der vorherigen roten Kerze – die Käufer haben die Kontrolle übernommen.",
    "Bearish Engulfing (bärisch)": "Eine rote Kerze schluckt komplett den Körper der vorherigen grünen Kerze – die Verkäufer haben die Kontrolle übernommen.",
    "Bullish Harami (bullisch)": "Eine kleine grüne Kerze liegt vollständig im Körper der vorherigen großen roten Kerze – mögliches Nachlassen des Verkaufsdrucks.",
    "Bearish Harami (bärisch)": "Eine kleine rote Kerze liegt vollständig im Körper der vorherigen großen grünen Kerze – mögliches Nachlassen des Kaufdrucks.",
    "Piercing Line (bullisch)": "Eine rote Kerze wird von einer grünen Kerze mehr als bis zur Körpermitte zurückerobert – mögliches bullisches Umkehrsignal.",
    "Dark Cloud Cover (bärisch)": "Eine rote Kerze fällt nach starkem Start unter die Körpermitte der vorherigen grünen Kerze – mögliches bearisches Umkehrsignal.",
    "Morning Star (bullisch)": "Dreier-Formation: große rote Kerze, kleine unentschlossene Kerze, dann eine große grüne Kerze – klassisches Bodenbildungsmuster.",
    "Evening Star (bärisch)": "Dreier-Formation: große grüne Kerze, kleine unentschlossene Kerze, dann eine große rote Kerze – klassisches Topbildungsmuster.",
    "3 weiße Soldaten (bullisch)": "Drei aufeinanderfolgende grüne Kerzen mit steigenden Schlusskursen – starker Aufwärtstrend.",
    "3 schwarze Krähen (bärisch)": "Drei aufeinanderfolgende rote Kerzen mit fallenden Schlusskursen – starker Abwärtstrend.",
    "Hammer (bullisch)": "Langer unterer Docht, kleiner Körper oben – Verkäufer drückten den Kurs runter, Käufer haben ihn zurückgeholt.",
    "Hanging Man (bärisch)": "Langer unterer Docht nach einem Anstieg – mögliches Warnsignal für nachlassende Käufer.",
    "Inverted Hammer (bullisch)": "Langer oberer Docht nach einem Abwärtstrend – mögliches erstes Zeichen für eine Bodenbildung.",
    "Shooting Star (bärisch)": "Langer oberer Docht, kleiner Körper unten – Käufer drückten den Kurs hoch, Verkäufer haben ihn zurückgeholt.",
    "Tweezer Bottom (bullisch)": "Zwei Kerzen testen nahezu dasselbe Tief, danach übernehmen die Käufer – mögliches Umkehrsignal.",
    "Tweezer Top (bärisch)": "Zwei Kerzen testen nahezu dasselbe Hoch, danach übernehmen die Verkäufer – mögliches Umkehrsignal.",
    "Doji (Unentschlossenheit)": "Open und Close liegen fast gleich – der Markt ist unentschlossen, oft ein Wendepunkt-Hinweis.",
    "Bullish Marubozu (bullisch)": "Eine lange grüne Kerze mit kaum Dochten zeigt starken, nahezu ununterbrochenen Kaufdruck.",
    "Bearish Marubozu (bärisch)": "Eine lange rote Kerze mit kaum Dochten zeigt starken, nahezu ununterbrochenen Verkaufsdruck.",
    "Spinning Top (Unentschlossenheit)": "Kleiner Körper und Dochte auf beiden Seiten zeigen ein ausgeglichenes Kräftemessen.",
    "Dragonfly Doji (bullisch)": "Open und Close liegen oben, während ein langer unterer Docht eine kräftige Erholung vom Tagestief zeigt.",
    "Gravestone Doji (bärisch)": "Open und Close liegen unten, während ein langer oberer Docht die Zurückweisung höherer Kurse zeigt.",
    "Bullish Belt Hold (bullisch)": "Eine starke grüne Kerze startet nahe dem Tief und schließt deutlich höher – bullischer Impuls.",
    "Bearish Belt Hold (bärisch)": "Eine starke rote Kerze startet nahe dem Hoch und schließt deutlich tiefer – bärischer Impuls.",
    "Three Inside Up (bullisch)": "Rote große Kerze, kleine Kerze innerhalb ihres Körpers und ein Ausbruch nach oben bilden eine mögliche Bodenwende.",
    "Three Inside Down (bärisch)": "Grüne große Kerze, kleine Kerze innerhalb ihres Körpers und ein Ausbruch nach unten bilden eine mögliche Topwende.",
    "Rising Three Methods (bullisch)": "Auf eine lange grüne Kerze folgt eine kurze Gegenbewegung innerhalb ihrer Spanne, danach setzt der Aufwärtstrend fort.",
    "Falling Three Methods (bärisch)": "Auf eine lange rote Kerze folgt eine kurze Gegenbewegung innerhalb ihrer Spanne, danach setzt der Abwärtstrend fort.",
    "Bullish Three Line Strike (bullisch)": "Drei steigende grüne Kerzen werden von einer großen roten Kerze zurückgenommen – mögliches Fortsetzungssignal im Aufwärtstrend.",
    "Bearish Three Line Strike (bärisch)": "Drei fallende rote Kerzen werden von einer großen grünen Kerze zurückgenommen – mögliches Fortsetzungssignal im Abwärtstrend.",
    "Bullish Kicker (bullisch)": "Ein starker Wechsel von einer roten zu einer grünen Kerze mit höherem Start signalisiert abrupten Kaufdruck.",
    "Bearish Kicker (bärisch)": "Ein starker Wechsel von einer grünen zu einer roten Kerze mit niedrigerem Start signalisiert abrupten Verkaufsdruck.",
}

# ------------------------------------------------------------
# Erweiterte Multi-Signal & Indikator-Erkennung
# ------------------------------------------------------------
def detect_pattern(df: pd.DataFrame) -> list[str]:
    if len(df) < 50:
        return ["Zu wenig Daten"]

    i = len(df) - 1
    found_signals = []

    for name, fn in PATTERNS.items():
        if fn(df, i):
            found_signals.append(name)

    close = df["Close"]
    volume = df["Volume"]

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_bb = sma20 + (std20 * 2)
    lower_bb = sma20 - (std20 * 2)

    if macd.iloc[i-1] < signal.iloc[i-1] and macd.iloc[i] > signal.iloc[i]:
        found_signals.append("MACD Bullish Crossover")
    elif macd.iloc[i-1] > signal.iloc[i-1] and macd.iloc[i] < signal.iloc[i]:
        found_signals.append("MACD Bearish Crossover")

    if rsi.iloc[i] < 30:
        found_signals.append("RSI Überverkauft (< 30)")
    elif rsi.iloc[i] > 70:
        found_signals.append("RSI Überhitzt (> 70)")

    if close.iloc[i] < lower_bb.iloc[i]:
        found_signals.append("Unter Bollinger-Band gefallen")
    elif close.iloc[i] > upper_bb.iloc[i]:
        found_signals.append("Über Bollinger-Band ausgebrochen")

    vol_mean = volume.rolling(20).mean()
    if len(volume) >= 20 and volume.iloc[i] > (1.8 * vol_mean.iloc[i]):
        found_signals.append("Hohes Handelsvolumen (Volumen-Spike)")

    if not found_signals:
        sma50 = close.rolling(50).mean()
        if len(close) >= 50 and close.iloc[i] > sma50.iloc[i]:
            found_signals.append("Neutrale Konsolidierung im Aufwärtstrend")
        else:
            found_signals.append("Neutrale Konsolidierung im Abwärtstrend")

    return found_signals

def historical_probability(df: pd.DataFrame, pattern: str) -> tuple[float, int]:
    if pattern not in PATTERNS:
        return None, 0
    fn = PATTERNS[pattern]
    hits, ups = 0, 0
    for i in range(len(df) - 1):
        if fn(df, i):
            hits += 1
            if df.iloc[i + 1]["Close"] > df.iloc[i]["Close"]:
                ups += 1
    if hits == 0:
        return None, 0
    return round(100 * ups / hits, 1), hits

# ------------------------------------------------------------
# Generalisierte Regel-Indikatoren (parametrisierbar für Grid-Search)
# ------------------------------------------------------------
DEFAULT_PARAMS = {
    "ema_fast": 9, "ema_slow": 21, "rsi_period": 14,
    "atr_period": 14, "atr_mult": 1.0, "reward_risk": 2.0,
    "rsi_bull": (50, 70), "rsi_bear": (30, 50),
}

def _trade_indicators(df: pd.DataFrame, ema_fast=9, ema_slow=21, rsi_period=14, atr_period=14) -> pd.DataFrame:
    data = df.copy()
    close = data["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, 1e-9)
    data["RSI"] = 100 - (100 / (1 + rs))
    data["EMA_FAST"] = close.ewm(span=ema_fast, adjust=False).mean()
    data["EMA_SLOW"] = close.ewm(span=ema_slow, adjust=False).mean()
    previous_close = close.shift(1)
    true_range = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - previous_close).abs(),
        (data["Low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    data["ATR"] = true_range.rolling(atr_period).mean()
    data["VolumeRatio"] = data["Volume"] / data["Volume"].rolling(20).mean().replace(0, np.nan)
    return data

def _trade_direction(row: pd.Series, rsi_bull=(50, 70), rsi_bear=(30, 50)) -> str:
    if pd.isna(row["RSI"]) or pd.isna(row["ATR"]):
        return "neutral"
    bullish = row["Close"] > row["EMA_FAST"] > row["EMA_SLOW"] and rsi_bull[0] <= row["RSI"] <= rsi_bull[1]
    bearish = row["Close"] < row["EMA_FAST"] < row["EMA_SLOW"] and rsi_bear[0] <= row["RSI"] <= rsi_bear[1]
    if bullish:
        return "long"
    if bearish:
        return "short"
    return "neutral"

def calculate_trade_setup(df: pd.DataFrame, params: dict = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    data = _trade_indicators(df, p["ema_fast"], p["ema_slow"], p["rsi_period"], p["atr_period"])
    row = data.iloc[-1]
    direction = _trade_direction(row, p["rsi_bull"], p["rsi_bear"])
    entry = float(row["Close"])
    atr = float(row["ATR"]) if not pd.isna(row["ATR"]) else 0.0
    stop_distance = atr * p["atr_mult"]
    if direction == "long":
        stop = entry - stop_distance
        target = entry + (p["reward_risk"] * stop_distance)
        signal = "KAUFEN (Long-Setup)"
    elif direction == "short":
        stop = entry + stop_distance
        target = entry - (p["reward_risk"] * stop_distance)
        signal = "VERKAUFEN (Short-Setup)"
    else:
        stop = target = None
        signal = "ABWARTEN"

    wins = losses = 0
    look_ahead = 5
    for index in range(len(data) - look_ahead):
        historical_row = data.iloc[index]
        historical_direction = _trade_direction(historical_row, p["rsi_bull"], p["rsi_bear"])
        historical_atr = historical_row["ATR"]
        if historical_direction == "neutral" or pd.isna(historical_atr) or historical_atr <= 0:
            continue
        historical_entry = float(historical_row["Close"])
        historical_stop_distance = float(historical_atr) * p["atr_mult"]
        if historical_direction == "long":
            historical_stop = historical_entry - historical_stop_distance
            historical_target = historical_entry + (p["reward_risk"] * historical_stop_distance)
        else:
            historical_stop = historical_entry + historical_stop_distance
            historical_target = historical_entry - (p["reward_risk"] * historical_stop_distance)
        for future_index in range(index + 1, index + look_ahead + 1):
            future_row = data.iloc[future_index]
            if historical_direction == "long":
                hit_stop = future_row["Low"] <= historical_stop
                hit_target = future_row["High"] >= historical_target
            else:
                hit_stop = future_row["High"] >= historical_stop
                hit_target = future_row["Low"] <= historical_target
            if hit_stop and hit_target:
                losses += 1
                break
            if hit_target:
                wins += 1
                break
            if hit_stop:
                losses += 1
                break

    total = wins + losses
    win_rate = round(100 * wins / total, 1) if total else None
    return {
        "data": data,
        "direction": direction,
        "signal": signal,
        "entry": entry,
        "stop": stop,
        "target": target,
        "atr": atr,
        "rsi": float(row["RSI"]) if not pd.isna(row["RSI"]) else None,
        "volume_ratio": float(row["VolumeRatio"]) if not pd.isna(row["VolumeRatio"]) else None,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
    }

def simulate_paper_bot(df: pd.DataFrame, initial_capital: float, risk_percent: float,
                        params: dict = None) -> dict:
    """Testet eine EMA/RSI/ATR-Regel ohne echte Orders oder Look-ahead."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    data = _trade_indicators(
        df, p["ema_fast"], p["ema_slow"], p["rsi_period"], p["atr_period"]
    ).reset_index(drop=True)
    training_end = max(50, int(len(data) * 0.7))
    equity = float(initial_capital)
    position = 0.0
    entry_price = stop_price = target_price = 0.0
    wins = losses = 0
    trades = []

    for index in range(training_end, len(data) - 1):
        row = data.iloc[index]
        next_row = data.iloc[index + 1]
        if position != 0:
            if position > 0:
                hit_stop = row["Low"] <= stop_price
                hit_target = row["High"] >= target_price
                exit_price = stop_price if hit_stop else target_price if hit_target else None
            else:
                hit_stop = row["High"] >= stop_price
                hit_target = row["Low"] <= target_price
                exit_price = stop_price if hit_stop else target_price if hit_target else None
            if exit_price is not None:
                pnl = position * (exit_price - entry_price)
                equity += pnl
                won = pnl > 0
                wins += int(won)
                losses += int(not won)
                trades.append({
                    "Datum": str(row["Date"]) if "Date" in row else str(index),
                    "Richtung": "Long" if position > 0 else "Short",
                    "Entry": round(entry_price, 4), "Exit": round(exit_price, 4),
                    "Ergebnis": round(pnl, 2),
                })
                position = 0.0
                continue

            direction = _trade_direction(row, p["rsi_bull"], p["rsi_bear"])
            if direction != "neutral" and ((position > 0 and direction == "short") or (position < 0 and direction == "long")):
                exit_price = float(next_row["Open"])
                pnl = position * (exit_price - entry_price)
                equity += pnl
                won = pnl > 0
                wins += int(won)
                losses += int(not won)
                trades.append({
                    "Datum": str(next_row["Date"]) if "Date" in next_row else str(index + 1),
                    "Richtung": "Long" if position > 0 else "Short",
                    "Entry": round(entry_price, 4), "Exit": round(exit_price, 4),
                    "Ergebnis": round(pnl, 2),
                })
                position = 0.0
            continue

        direction = _trade_direction(row, p["rsi_bull"], p["rsi_bear"])
        atr = row["ATR"]
        if direction == "neutral" or pd.isna(atr) or atr <= 0 or equity <= 0:
            continue
        entry_price = float(next_row["Open"])
        stop_distance = float(atr) * p["atr_mult"]
        risk_amount = equity * risk_percent / 100
        units = risk_amount / stop_distance if stop_distance else 0
        if direction == "long":
            position = units
            stop_price = entry_price - stop_distance
            target_price = entry_price + p["reward_risk"] * stop_distance
        else:
            position = -units
            stop_price = entry_price + stop_distance
            target_price = entry_price - p["reward_risk"] * stop_distance

    if position != 0:
        exit_price = float(data.iloc[-1]["Close"])
        pnl = position * (exit_price - entry_price)
        equity += pnl
        wins += int(pnl > 0)
        losses += int(pnl <= 0)
        trades.append({
            "Datum": str(data.iloc[-1]["Date"]) if "Date" in data else str(len(data) - 1),
            "Richtung": "Long" if position > 0 else "Short",
            "Entry": round(entry_price, 4), "Exit": round(exit_price, 4),
            "Ergebnis": round(pnl, 2),
        })

    total = wins + losses
    return {
        "params": p,
        "training_end": training_end,
        "final_equity": equity,
        "return_percent": (equity / initial_capital - 1) * 100,
        "wins": wins, "losses": losses,
        "win_rate": 100 * wins / total if total else None,
        "trade_count": total,
        "trades": trades,
    }

def run_live_paper_check(df: pd.DataFrame, ticker: str, account: dict) -> dict:
    """Process one new candle for a long-only, no-money paper account."""
    setup = calculate_trade_setup(df)
    candle_time = str(df.index[-1])
    price = float(setup["entry"])
    position = account.get("position")
    event = None

    if position and position.get("ticker") == ticker:
        close_reason = None
        if price <= position["stop"]:
            close_reason = "Stop-Loss erreicht"
        elif price >= position["target"]:
            close_reason = "Take-Profit erreicht"
        elif setup["direction"] == "short":
            close_reason = "Trendwechsel: EMA/RSI geben ein Verkaufssignal"
        if close_reason:
            proceeds = position["units"] * price
            account["cash"] += proceeds
            pnl = proceeds - position["cost"]
            event = {
                "Zeit": candle_time, "Aktion": "VERKAUF", "Asset": ticker,
                "Preis": round(price, 4), "Menge": round(position["units"], 6),
                "Ergebnis": round(pnl, 2), "Warum": close_reason,
            }
            account["position"] = None
    elif not position and account.get("last_candle") != candle_time and setup["direction"] == "long" and setup["stop"] is not None:
        risk_per_unit = max(price - float(setup["stop"]), 1e-9)
        risk_budget = account["cash"] * account["risk_percent"] / 100
        units = min(risk_budget / risk_per_unit, account["cash"] / price)
        if units > 0:
            cost = units * price
            account["cash"] -= cost
            account["position"] = {
                "ticker": ticker, "units": units, "cost": cost, "stop": float(setup["stop"]),
                "target": float(setup["target"]),
            }
            event = {
                "Zeit": candle_time, "Aktion": "KAUF", "Asset": ticker,
                "Preis": round(price, 4), "Menge": round(units, 6), "Ergebnis": 0.0,
                "Warum": "Long-Signal: Kurs über EMA 9/21 und RSI im bullischen Bereich",
            }

    account["last_candle"] = candle_time
    if event:
        account["events"].append(event)
    position_value = (account["position"] or {}).get("units", 0) * price
    account["equity"] = account["cash"] + position_value
    account["last_reason"] = event["Warum"] if event else "Keine Order: aktuelles Regelwerk liefert kein neues Long- oder Ausstiegssignal."
    account["last_setup"] = setup
    return account

def load_scanner_account(starting_cash: float, risk_percent: float) -> tuple[dict | None, list[dict]]:
    """Load the single server-managed scanner account and its recent orders."""
    client = get_supabase_client()
    if client is None:
        return None, []
    try:
        result = client.table("scanner_paper_accounts").select("*").eq("account_key", SCANNER_ACCOUNT_KEY).maybe_single().execute()
        row = result.data
        if not row:
            client.table("scanner_paper_accounts").insert({
                "account_key": SCANNER_ACCOUNT_KEY, "cash": starting_cash,
                "equity": starting_cash, "risk_percent": risk_percent,
            }).execute()
            row = client.table("scanner_paper_accounts").select("*").eq("account_key", SCANNER_ACCOUNT_KEY).maybe_single().execute().data
            if not row:
                return None, []
        account = {
            "cash": float(row["cash"]), "equity": float(row["equity"]),
            "risk_percent": float(row["risk_percent"]), "position": row.get("position"),
            "last_candle": row.get("last_candle"), "events": [],
        }
        orders = client.table("scanner_paper_orders").select("created_at,action,ticker,price,units,pnl,reason").eq("account_key", SCANNER_ACCOUNT_KEY).order("created_at", desc=True).limit(50).execute().data or []
        return account, orders
    except Exception as error:
        st.session_state["_supabase_last_error"] = f"Supabase-Abfrage fehlgeschlagen: {error!r}"
        return None, []

def save_scanner_account(account: dict, event: dict | None) -> None:
    client = get_supabase_client()
    if client is None:
        return
    client.table("scanner_paper_accounts").update({
        "cash": account["cash"], "equity": account["equity"],
        "risk_percent": account["risk_percent"], "position": account["position"],
        "last_candle": account.get("last_candle"), "updated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
    }).eq("account_key", SCANNER_ACCOUNT_KEY).execute()
    if event:
        client.table("scanner_paper_orders").insert({
            "account_key": SCANNER_ACCOUNT_KEY,
            "action": "BUY" if event["Aktion"] == "KAUF" else "SELL",
            "ticker": event["Asset"], "price": event["Preis"], "units": event["Menge"],
            "pnl": event["Ergebnis"], "reason": event["Warum"],
            "signal": {"source": "ema_rsi_atr_scanner"},
        }).execute()

def scan_and_trade_paper_market(account: dict, interval_key: str, limit: int) -> tuple[dict, pd.DataFrame]:
    """Scan a capped liquid universe and paper-trade only the best valid setup."""
    candidates = []
    universe = list(SCANNER_UNIVERSE.items())[:limit]
    active_ticker = (account.get("position") or {}).get("ticker")
    if active_ticker:
        universe = [(label, ticker) for label, ticker in SCANNER_UNIVERSE.items() if ticker == active_ticker]

    for label, ticker in universe:
        try:
            data = load_data(ticker, interval_key, "Yahoo Finance")
            if data.empty or len(data) < 50:
                continue
            setup = calculate_trade_setup(data)
            volume_score = min(float(setup.get("volume_ratio") or 0), 3.0)
            rsi = float(setup.get("rsi") or 0)
            score = volume_score * 10 + (70 - abs(60 - rsi))
            candidates.append({
                "Asset": label, "Ticker": ticker, "Signal": setup["signal"],
                "Score": round(score, 1), "RSI": round(rsi, 1),
                "Volumen": round(volume_score, 2), "_data": data,
            })
        except Exception:
            continue

    visible = pd.DataFrame([{k: v for k, v in item.items() if k != "_data"} for item in candidates])
    if not candidates:
        account["last_reason"] = "Scanner konnte für das gewählte Universum keine ausreichenden Marktdaten laden."
        return account, visible
    candidates.sort(key=lambda item: item["Score"], reverse=True)
    tradable = [item for item in candidates if item["Signal"].startswith("KAUFEN")]
    chosen = tradable[0] if tradable else candidates[0]
    before = len(account["events"])
    account = run_live_paper_check(chosen["_data"], chosen["Ticker"], account)
    event = account["events"][-1] if len(account["events"]) > before else None
    save_scanner_account(account, event)
    return account, visible.sort_values("Score", ascending=False)

def optimize_paper_bot(df: pd.DataFrame, initial_capital: float, risk_percent: float,
                        min_trades: int = 8, max_tests: int | None = None) -> dict:
    """
    Grid-Search über EMA/RSI/ATR/Chance-Risiko-Kombinationen.
    Bewertet wird ausschließlich auf einem Out-of-Sample-Holdout-Zeitraum,
    damit sich die Auswahl nicht einfach an die Vergangenheit anpasst (Overfitting).
    """
    grid = {
        "ema_fast": [5, 9, 12],
        "ema_slow": [21, 26, 34],
        "rsi_period": [10, 14, 21],
        "atr_mult": [1.0, 1.5, 2.0],
        "reward_risk": [1.5, 2.0, 3.0],
    }
    keys = list(grid.keys())
    combos = [
        dict(zip(keys, combo))
        for combo in itertools.product(*grid.values())
        if combo[0] < combo[1]
    ]

    # Evenly spread a smaller test budget over the entire parameter space
    # instead of only checking the first combinations in the grid.
    if max_tests is not None and max_tests < len(combos):
        selected_indices = np.linspace(0, len(combos) - 1, max_tests, dtype=int)
        combos = [combos[index] for index in selected_indices]

    split = int(len(df) * 0.85)
    fit_df = df.iloc[:split].copy()
    holdout_df = df.iloc[max(0, split - 60):].copy()

    best = None
    tests_run = 0
    for base_params in combos:
        params = dict(base_params)
        params["rsi_bull"] = (50, 70)
        params["rsi_bear"] = (30, 50)
        params["atr_period"] = 14

        tests_run += 1
        fit_result = simulate_paper_bot(fit_df, initial_capital, risk_percent, params)
        if fit_result["trade_count"] < min_trades:
            continue

        holdout_result = simulate_paper_bot(holdout_df, initial_capital, risk_percent, params)
        if holdout_result["trade_count"] < 3:
            continue

        score = holdout_result["return_percent"]
        if best is None or score > best["score"]:
            best = {
                "params": params, "score": score, "fit": fit_result,
                "holdout": holdout_result, "tests_run": tests_run,
            }

    if best is not None:
        best["tests_run"] = tests_run
    return best

# ------------------------------------------------------------
# Lernendes ML-Modell (Gradient Boosting, Walk-Forward-Validierung)
# ------------------------------------------------------------
ML_HORIZON = 5  # Kerzen in die Zukunft, deren Richtung vorhergesagt wird

def build_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    close = data["Close"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    data["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data["bb_pos"] = (close - sma20) / std20.replace(0, np.nan)

    data["ema_fast"] = close.ewm(span=9, adjust=False).mean()
    data["ema_slow"] = close.ewm(span=21, adjust=False).mean()
    data["ema_gap"] = (data["ema_fast"] - data["ema_slow"]) / close

    data["ret_1"] = close.pct_change(1)
    data["ret_5"] = close.pct_change(5)
    data["ret_10"] = close.pct_change(10)

    vol_mean = data["Volume"].rolling(20).mean()
    data["vol_ratio"] = data["Volume"] / vol_mean.replace(0, np.nan)

    body = (data["Close"] - data["Open"]).abs()
    rng = (data["High"] - data["Low"]).replace(0, np.nan)
    data["body_ratio"] = body / rng

    future_return = close.shift(-ML_HORIZON) / close - 1
    threshold = data["ret_1"].rolling(50).std().fillna(data["ret_1"].std()) * 1.0
    data["future_return"] = future_return
    data["target"] = np.where(
        future_return > threshold, 1,
        np.where(future_return < -threshold, 0, np.nan),
    )

    feature_cols = [
        "rsi", "macd_hist", "bb_pos", "ema_gap",
        "ret_1", "ret_5", "ret_10", "vol_ratio", "body_ratio",
    ]
    return data, feature_cols

def train_walkforward_ml(df: pd.DataFrame, n_folds: int = 5) -> dict:
    """
    Walk-Forward-Validierung: Das Modell wird immer nur auf der Vergangenheit
    trainiert und auf dem direkt folgenden, ihm unbekannten Abschnitt getestet.
    So bekommst du eine ehrliche Einschätzung, ob das Modell wirklich etwas
    "gelernt" hat, statt nur die Vergangenheit auswendig zu kennen.
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn ist nicht installiert. Bitte 'pip install scikit-learn' ausführen."}

    data, feature_cols = build_ml_features(df)
    usable = data.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    if len(usable) < 200:
        return {"error": "Zu wenig nutzbare Datenpunkte für ein Walk-Forward-Training (mind. ~200 benötigt)."}

    fold_size = len(usable) // (n_folds + 1)
    fold_accuracies = []
    fold_details = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_end = fold_size * (fold + 2)
        train_slice = usable.iloc[:train_end]
        test_slice = usable.iloc[train_end:test_end]
        if len(train_slice) < 50 or len(test_slice) < 10:
            continue

        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42,
        )
        model.fit(train_slice[feature_cols], train_slice["target"])
        predictions = model.predict(test_slice[feature_cols])
        accuracy = float((predictions == test_slice["target"].values).mean())
        fold_accuracies.append(accuracy)
        fold_details.append({
            "fold": fold + 1,
            "train_size": len(train_slice),
            "test_size": len(test_slice),
            "accuracy": round(accuracy * 100, 1),
        })

    if not fold_accuracies:
        return {"error": "Nicht genug Daten, um Walk-Forward-Folds zu bilden. Größeres Intervall oder mehr Historie wählen."}

    final_model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42,
    )
    final_model.fit(usable[feature_cols], usable["target"])

    last_row = data.iloc[[-1]][feature_cols]
    if last_row.isna().any(axis=1).iloc[0]:
        live_probability = None
    else:
        live_probability = float(final_model.predict_proba(last_row)[0][1])

    baseline_up_rate = float(usable["target"].mean())

    return {
        "mean_accuracy": round(100 * float(np.mean(fold_accuracies)), 1),
        "fold_details": fold_details,
        "baseline_up_rate": round(100 * baseline_up_rate, 1),
        "live_probability_up": round(100 * live_probability, 1) if live_probability is not None else None,
        "feature_importances": dict(zip(feature_cols, final_model.feature_importances_.round(3))),
        "horizon": ML_HORIZON,
        "sample_size": len(usable),
    }

def render_ml_predictor():
    with st.expander("KI-Prognose (gemeinsames lernendes Modell)", expanded=False):
        st.caption(
            "Das Modell sammelt aus den Paper-Analysen gemeinsame Trainingsbeispiele in Supabase. "
            "Neue Modelle werden per Walk-Forward validiert und nur übernommen, wenn sie das aktive Modell nicht verschlechtern."
        )
        status = get_learning_status()
        if not status["ready"]:
            st.warning("Gemeinsames Lernen ist noch nicht verbunden. Setze die Supabase-Secrets, bevor du die Web-App veröffentlichst.")
        else:
            a, b, c = st.columns(3)
            a.metric("Gemeinsame Beispiele", str(status["examples"]))
            b.metric("Aktives Modell", "Ja" if status["active_model"] else "Noch keins")
            c.metric("Letztes Training", str(status["last_training"] or "Noch keins")[:19])

        if not SKLEARN_AVAILABLE:
            st.warning("Für dieses Feature fehlt scikit-learn.")
            return

        ml_options = list(ASSETS.keys()) + st.session_state.watchlist
        ml_asset = st.selectbox("Asset", ml_options, key="ml_asset")
        ml_interval = st.selectbox(
            "Intervall", list(INTERVAL_CONFIG.keys()),
            index=list(INTERVAL_CONFIG.keys()).index("1d"), key="ml_interval",
        )

        if st.button("Daten sammeln & gemeinsames Modell trainieren", key="run_ml"):
            ticker = ASSETS.get(ml_asset) or ml_asset
            with st.spinner("Sammle Paper-Daten und prüfe das gemeinsame Modell..."):
                try:
                    ml_df = load_data(ticker, ml_interval, "Yahoo Finance")
                    if ml_df.empty or len(ml_df) < 250:
                        st.error("Für sinnvolles Training werden mindestens 250 Kerzen benötigt.")
                    else:
                        feature_df, feature_cols = build_ml_features(ml_df)
                        saved = save_learning_examples(ticker, ml_interval, feature_df, feature_cols)
                        st.info(f"{saved} Trainingsbeispiele aus {ticker} wurden synchronisiert.")
                        status_after = get_learning_status()
                        if status_after["examples"] >= TRAINING_MIN_SAMPLES:
                            result = train_and_maybe_promote_shared_model()
                            st.session_state.ml_result = result
                        else:
                            st.session_state.ml_result = {"error": f"Noch {TRAINING_MIN_SAMPLES - status_after['examples']} Beispiele bis zum ersten gemeinsamen Training."}
                except Exception as exc:
                    st.session_state.ml_result = {"error": str(exc)}

        result = st.session_state.ml_result
        if result is not None:
            if "error" in result:
                st.warning(result["error"])
            elif result.get("ok"):
                metrics = result["metrics"]
                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Walk-Forward-Trefferquote", f'{metrics["mean_accuracy"]}%')
                metric_b.metric("Naive Basislinie", f'{metrics["baseline_accuracy"]}%')
                metric_c.metric("Modellstatus", "Übernommen" if result["promoted"] else "Verworfen")
                if result.get("previous_accuracy") is not None:
                    st.caption(f'Vorheriges aktives Modell: {float(result["previous_accuracy"]):.2f}% Walk-Forward-Trefferquote.')
                st.dataframe(pd.DataFrame(metrics["fold_details"]), use_container_width=True, hide_index=True)
                importance_df = pd.DataFrame(
                    sorted(metrics["feature_importances"].items(), key=lambda kv: kv[1], reverse=True),
                    columns=["Feature", "Gewichtung"],
                )
                st.dataframe(importance_df, use_container_width=True, hide_index=True)
                st.caption("Das System ist weiterhin Paper-Trading/Analyse. Historische Modelltests garantieren keine zukünftigen Ergebnisse.")


def analyze_ticker(ticker: str, interval_key: str = "1d", source: str = "Yahoo Finance"):
    try:
        df = load_data(ticker, interval_key, source)
    except Exception:
        return None
    if df.empty or len(df) < 20:
        return None

    # Every analysis enriches the same anonymized, shared market-data pool.
    # Labels are derived solely from public price candles, never from a user.
    collect_shared_learning(ticker, interval_key, df)

    patterns_list = detect_pattern(df)
    pattern_str = ", ".join(patterns_list)

    probability, hits = None, 0
    for p in patterns_list:
        if p in PATTERNS:
            prob, h = historical_probability(df, p)
            if prob is not None:
                probability, hits = prob, h
                break

    return {
        "ticker": ticker, "df": df, "pattern": pattern_str,
        "patterns_list": patterns_list, "probability": probability, "hits": hits,
        "trade_setup": calculate_trade_setup(df),
    }

# ------------------------------------------------------------
# Candlestick-Chart (ohne Nacht-Lücken & mit Zoom-Reset)
# ------------------------------------------------------------
def render_candlestick_chart(df: pd.DataFrame, pattern: str, ticker: str, n_candles: int = 40):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    plot_df = df.tail(n_candles).copy().reset_index(drop=True)

    if pd.api.types.is_datetime64_any_dtype(plot_df["Date"]):
        dates = pd.to_datetime(plot_df["Date"])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_convert("Europe/Berlin")
        has_intraday_time = bool((dates.dt.hour != 0).any() or (dates.dt.minute != 0).any())
        plot_df["DateStr"] = dates.dt.strftime("%d.%m. %H:%M" if has_intraday_time else "%d.%m.%Y")
    else:
        plot_df["DateStr"] = plot_df["Date"].astype(str)

    fig = go.Figure(data=[go.Candlestick(
        x=plot_df["DateStr"],
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"],
        increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
        decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
        name=ticker,
    )])

    if pattern and "Neutrale" not in pattern:
        last_row = plot_df.iloc[-1]
        h = last_row["High"]
        short_pattern = pattern.split(", ")[0]
        if len(short_pattern) > 28:
            short_pattern = short_pattern[:25] + "..."

        fig.add_annotation(
            x=last_row["DateStr"], y=h,
            text=short_pattern, showarrow=True, arrowhead=2, arrowcolor="#ffd60a",
            font=dict(color="#ffd60a", size=10), bgcolor="#30384a",
            bordercolor="#ffd60a", borderwidth=1, borderpad=4,
            xanchor="right", ax=-8, ay=-42,
        )

    tick_step = max(1, len(plot_df) // 6)
    tick_values = plot_df["DateStr"].iloc[::tick_step].tolist()
    fig.update_layout(
        paper_bgcolor="#252b39", plot_bgcolor="#252b39",
        font=dict(color="#e0e5ef"),
        margin=dict(l=12, r=12, t=58, b=54),
        height=340,
        xaxis=dict(
            type="category",
            showgrid=False,
            tickmode="array",
            tickvals=tick_values,
            tickangle=-30,
            tickfont=dict(size=10, color="#b8c2d3"),
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(showgrid=True, gridcolor="#3b4354", tickfont=dict(color="#b8c2d3")),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Gemini-Analyse
# ------------------------------------------------------------
def get_gemini_analysis(api_key: str, asset: str, pattern: str, probability, df: pd.DataFrame) -> str:
    last_close = round(float(df.iloc[-1]["Close"]), 2)
    cache_key = f"{asset}|{pattern}|{probability}|{last_close}"
    if cache_key in st.session_state.ai_cache:
        return st.session_state.ai_cache[cache_key]

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "⚠️ Paket 'google-genai' fehlt. Bitte requirements.txt prüfen."

    try:
        client = genai.Client(api_key=api_key)

        close = df["Close"]
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.mean()
        current_price = float(close.iloc[-1])

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        rsi = round(float(rsi_series.iloc[-1]), 1) if not np.isnan(rsi_series.iloc[-1]) else 50.0

        trend = "Aufwärtstrend (über 200-Tage-Linie)" if current_price > sma200 else "Abwärtstrend (unter 200-Tage-Linie)"
        last_candles = df.tail(3)[["Open", "High", "Low", "Close"]].round(1).values.tolist()
        prob_str = f"{probability}%" if probability is not None else "Kein historischer Vorteil"

        prompt = (
            f"Analysiere {asset} als Trading-Experte:\n"
            f"- Aktueller Kurs: {round(current_price, 2)}\n"
            f"- Übergeordneter Trend: {trend}\n"
            f"- RSI (14): {rsi} (Unter 30 = überverkauft/Einstiegschance, Über 70 = überhitzt/Korrekturgefahr)\n"
            f"- Aktive Muster & Signale: {pattern} (Trefferquote: {prob_str})\n"
            f"- Letzte 3 Kerzen (OHLC): {last_candles}\n\n"
            f"Antworte auf Deutsch in GENAU diesem Format (4 Zeilen, je 1 kurzer Satz):\n"
            f"1) EMPFEHLUNG: Kaufen / Verkaufen / Abwarten – mit kurzem Warum\n"
            f"2) BEGRÜNDUNG: Kombiniere Trend, RSI und Signale\n"
            f"3) CHANCE: Chance auf Plus: X% / Chance auf Minus: Y% (X+Y=100)\n"
            f"4) RISIKO: Kurzer Risikohinweis"
        )

        try:
            config = types.GenerateContentConfig(
                max_output_tokens=500,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        except Exception:
            config = types.GenerateContentConfig(max_output_tokens=500)

        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash", contents=prompt, config=config,
            )
            text = (response.text or "").strip()
            if not text:
                return "Gemini hat keine Antwort geliefert. Die mathematische Analyse oben bleibt verfügbar."
            st.session_state.ai_cache[cache_key] = text
            return text
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                return "Gemini-Limit erreicht. Bitte später erneut auf KI-Einschätzung laden klicken."
            if "UNAVAILABLE" in err or "503" in err:
                return "Gemini ist gerade nicht verfügbar. Die mathematische Analyse oben bleibt verfügbar."
            raise

    except Exception as e:
        return f"⚠️ Gemini-Fehler: {e}"

def get_ai_analysis(provider: str, api_key: str, asset: str, pattern: str, probability, df: pd.DataFrame) -> str:
    if provider == "Gemini":
        return get_gemini_analysis(api_key, asset, pattern, probability, df)

    last_close = round(float(df.iloc[-1]["Close"]), 2)
    cache_key = f"{provider}|{asset}|{pattern}|{probability}|{last_close}"
    if cache_key in st.session_state.ai_cache:
        return st.session_state.ai_cache[cache_key]

    close = df["Close"]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.mean()
    current_price = float(close.iloc[-1])
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = round(float(rsi_series.iloc[-1]), 1) if not np.isnan(rsi_series.iloc[-1]) else 50.0
    trend = "Aufwärtstrend" if current_price > sma200 else "Abwärtstrend"
    prompt = (
        f"Analysiere {asset} kurz auf Deutsch. Kurs {current_price:.2f}, Trend {trend}, "
        f"RSI {rsi}, Signale {pattern}, historische Chance {probability if probability is not None else 'unbekannt'}%. "
        "Antworte in genau 3 kurzen Zeilen: Empfehlung (Kaufen/Verkaufen/Abwarten), "
        "Begründung, Risiko. Keine Anlageberatung."
    )
    try:
        if provider == "OpenAI":
            payload = json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 180,
                "temperature": 0.2,
            }).encode("utf-8")
            request = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions", data=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                text = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()
        else:
            payload = json.dumps({
                "model": "claude-3-5-haiku-latest",
                "max_tokens": 180,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            }).encode("utf-8")
            request = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=payload,
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                text = json.loads(response.read().decode("utf-8"))["content"][0]["text"].strip()
        st.session_state.ai_cache[cache_key] = text
        return text
    except urllib.error.HTTPError as error:
        if error.code in (429, 529):
            return f"{provider}-Limit erreicht. Bitte später erneut versuchen."
        return f"{provider}-Fehler ({error.code}). Die mathematische Analyse bleibt verfügbar."
    except Exception as error:
        return f"{provider}-Fehler: {error}"

def render_result_card(ticker: str, pattern: str, probability, hits: int, n_candles: int, interval_key: str):
    if probability is None:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="pattern-name">{ticker} · {pattern}</div>
                <div class="pattern-meta">Erkannte Signale · letzte {n_candles} Kerzen ({interval_key})</div>
                <div class="prob-box neutral-box">Aktiviertes Setup</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    is_bullish = probability >= 50
    box_class = "green-box" if is_bullish else "red-box"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="pattern-name">{ticker} · {pattern}</div>
            <div class="pattern-meta">{hits} historische Vergleichsfälle für Kerzenmuster · letzte {n_candles} Kerzen ({interval_key}) · Chance auf Plus</div>
            <div class="prob-box {box_class}">{probability}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_trade_setup(setup: dict, ticker: str, interval_key: str):
    st.markdown('<div class="section-label">Daytrading-Setup</div>', unsafe_allow_html=True)
    st.caption("Marktdaten von Yahoo Finance: je nach Börse und Intervall möglicherweise verzögert, kein garantierter Tick-Livefeed. Vor einer Order bitte den Brokerkurs prüfen.")

    signal_color = "#26a69a" if setup["direction"] == "long" else "#ef5350" if setup["direction"] == "short" else "#ffd60a"
    win_loss = f'{setup["wins"]} W / {setup["losses"]} L'
    win_rate = f'{setup["win_rate"]}%' if setup["win_rate"] is not None else "Nicht genug Fälle"
    rsi = f'{setup["rsi"]:.1f}' if setup["rsi"] is not None else "–"
    volume_ratio = f'{setup["volume_ratio"]:.2f}x' if setup["volume_ratio"] is not None else "–"
    entry_value = f'{setup["entry"]:.2f}'
    stop_value = f'{setup["stop"]:.2f}' if setup["stop"] is not None else "–"
    target_value = f'{setup["target"]:.2f}' if setup["target"] is not None else "–"
    levels = "Kein aktives Setup: Stop-Loss und Take-Profit werden erst bei einem klaren Long- oder Short-Signal berechnet."
    if setup["direction"] != "neutral":
        levels = "Stop-Loss begrenzt den geplanten Verlust; Take-Profit ist das automatisch berechnete Kursziel bei einem Chance-Risiko-Verhältnis von 2:1."

    st.markdown(
        f'<div class="trade-card">'
        f'<div class="trade-title">{ticker} · {interval_key}</div>'
        f'<div class="trade-subtitle">Regelbasiert aus EMA/RSI/ATR und Volumen</div>'
        f'<div class="trade-signal" style="color:{signal_color};">{setup["signal"]}</div>'
        f'<div class="trade-grid">'
        f'<div class="trade-metric"><span class="trade-metric-label">Entry / aktueller Kurs</span><span class="trade-metric-value">{entry_value}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Stop-Loss</span><span class="trade-metric-value">{stop_value}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Take-Profit / Kursziel</span><span class="trade-metric-value">{target_value}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">RSI (0–100)</span><span class="trade-metric-value">{rsi}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Volumen vs. 20er-Schnitt</span><span class="trade-metric-value">{volume_ratio}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Historische Trefferquote</span><span class="trade-metric-value">{win_rate} · {win_loss}</span></div>'
        f'</div><div class="trade-note">{levels}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Risiko- und Hebel-Rechner</div>', unsafe_allow_html=True)
    capital_col, risk_col, leverage_col = st.columns(3)
    with capital_col:
        capital = st.number_input("Kapital", min_value=50.0, value=1000.0, step=50.0, key=f"capital_{ticker}_{interval_key}")
    with risk_col:
        risk_percent = st.number_input("Risiko %", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key=f"risk_{ticker}_{interval_key}")
    with leverage_col:
        leverage = st.number_input("Hebel", min_value=1.0, max_value=10.0, value=1.0, step=0.5, key=f"leverage_{ticker}_{interval_key}")

    calculation_mode = st.selectbox(
        "Rechenrichtung",
        ["Automatisch (Signal)", "Long berechnen", "Short berechnen"],
        key=f"calculation_mode_{ticker}_{interval_key}",
        help="Automatisch verwendet nur ein erkanntes Setup. Long/Short berechnen ist eine separate Beispielrechnung und keine Empfehlung.",
    )
    calculation_direction = setup["direction"]
    if calculation_mode == "Long berechnen":
        calculation_direction = "long"
    elif calculation_mode == "Short berechnen":
        calculation_direction = "short"

    if calculation_direction == "neutral":
        st.info("Das aktuelle Marktsignal lautet Abwarten. Wähle oben Long oder Short berechnen, um nur die Positionsgröße zu simulieren.")
        return

    calculation_stop = setup["entry"] - setup["atr"] if calculation_direction == "long" else setup["entry"] + setup["atr"]
    calculation_target = setup["entry"] + (2 * setup["atr"]) if calculation_direction == "long" else setup["entry"] - (2 * setup["atr"])
    if calculation_mode != "Automatisch (Signal)":
        st.caption(f"Beispielrechnung: Entry {setup['entry']:.2f} · Stop-Loss {calculation_stop:.2f} · Take-Profit {calculation_target:.2f}")

    risk_per_unit = abs(setup["entry"] - calculation_stop)
    risk_amount = capital * risk_percent / 100
    risk_based_units = risk_amount / risk_per_unit if risk_per_unit else 0
    margin_limited_units = capital * leverage / setup["entry"] if setup["entry"] else 0
    units = min(risk_based_units, margin_limited_units)
    notional = units * setup["entry"]
    margin = notional / leverage
    actual_risk = units * risk_per_unit
    st.markdown(
        f'<div class="trade-card"><div class="trade-grid">'
        f'<div class="trade-metric"><span class="trade-metric-label">Max. Stückzahl (Einheiten)</span><span class="trade-metric-value">{units:.4f}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Positionswert</span><span class="trade-metric-value">{notional:.2f}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Gebundene Margin</span><span class="trade-metric-value">{margin:.2f}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Max. Verlust am Stop (€)</span><span class="trade-metric-value">{actual_risk:.2f} ({actual_risk / capital * 100:.2f}%)</span></div>'
        f'</div><div class="trade-note">Max. Stückzahl bedeutet: so viele Einheiten können bis zum Stop gehalten werden. Risiko % wird zuerst in Euro umgerechnet. Gebühren, Slippage, Finanzierungskosten und Gaps sind nicht eingerechnet.</div></div>',
        unsafe_allow_html=True,
    )

def render_mini_card(ticker: str, pattern: str, probability):
    if probability is None:
        st.markdown(
            f"""
            <div class="mini-card">
                <div>
                    <div class="mini-ticker">{ticker}</div>
                    <div class="mini-pattern">{pattern}</div>
                </div>
                <div class="mini-prob" style="color:#8e8e93; background:#2c2c2e;">–</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    is_bullish = probability >= 50
    color = "#26a69a" if is_bullish else "#ef5350"
    bg = "rgba(38,166,154,0.14)" if is_bullish else "rgba(239,83,80,0.14)"
    st.markdown(
        f"""
        <div class="mini-card">
            <div>
                <div class="mini-ticker">{ticker}</div>
                <div class="mini-pattern">{pattern}</div>
            </div>
            <div class="mini-prob" style="color:{color}; background:{bg};">{probability}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# Candlestick-Lehrbuch
# ------------------------------------------------------------
def render_pattern_text(pattern: str) -> str:
    return PATTERN_EXPLAIN.get(pattern, "Dieses Muster beschreibt eine mögliche Veränderung des Kauf- und Verkaufsdrucks.")

def render_pattern_visual(pattern: str) -> str:
    single_patterns = {"Hammer (bullisch)", "Hanging Man (bärisch)", "Inverted Hammer (bullisch)", "Shooting Star (bärisch)", "Doji (Unentschlossenheit)", "Bullish Marubozu (bullisch)", "Bearish Marubozu (bärisch)", "Spinning Top (Unentschlossenheit)", "Dragonfly Doji (bullisch)", "Gravestone Doji (bärisch)", "Bullish Belt Hold (bullisch)", "Bearish Belt Hold (bärisch)"}
    three_patterns = {"Morning Star (bullisch)", "Evening Star (bärisch)", "3 weiße Soldaten (bullisch)", "3 schwarze Krähen (bärisch)", "Three Inside Up (bullisch)", "Three Inside Down (bärisch)"}
    if pattern == "Doji (Unentschlossenheit)":
        candles = ["doji"]
    elif pattern == "Dragonfly Doji (bullisch)":
        candles = ["dragonfly"]
    elif pattern == "Gravestone Doji (bärisch)":
        candles = ["gravestone"]
    elif pattern == "Spinning Top (Unentschlossenheit)":
        candles = ["doji"]
    elif pattern in single_patterns:
        candles = ["bear long" if any(word in pattern for word in ["Bearish", "Hanging", "Shooting"]) else "bull long"]
    elif pattern in three_patterns:
        if "Morning" in pattern:
            candles = ["bear long", "bull small", "bull long"]
        elif "Evening" in pattern:
            candles = ["bull long", "bull small", "bear long"]
        elif "Inside Up" in pattern:
            candles = ["bear long", "bull small", "bull long"]
        elif "Inside Down" in pattern:
            candles = ["bull long", "bear small", "bear long"]
        elif "Soldaten" in pattern:
            candles = ["bull", "bull", "bull"]
        else:
            candles = ["bear", "bear", "bear"]
    elif "Rising Three" in pattern:
        candles = ["bull long", "bear small", "bear small", "bear small", "bull long"]
    elif "Falling Three" in pattern:
        candles = ["bear long", "bull small", "bull small", "bull small", "bear long"]
    elif "Three Line Strike" in pattern:
        candles = ["bear", "bear", "bear", "bull long"] if "Bullish" in pattern else ["bull", "bull", "bull", "bear long"]
    elif "Kicker" in pattern:
        candles = ["bear", "bull long"] if "Bullish" in pattern else ["bull", "bear long"]
    elif "Harami" in pattern:
        candles = ["bear long", "bull small"] if "Bullish" in pattern else ["bull long", "bear small"]
    elif "Engulfing" in pattern:
        candles = ["bear small", "bull long"] if "Bullish" in pattern else ["bull small", "bear long"]
    elif "Piercing" in pattern:
        candles = ["bear long", "bull"]
    elif "Dark Cloud" in pattern:
        candles = ["bull long", "bear"]
    elif "Tweezer Bottom" in pattern:
        candles = ["bear", "bull"]
    else:
        candles = ["bull", "bear"]
    candle_html = "".join(
        f'<div class="candle {candle}"><span class="candle-wick"></span><span class="candle-body"></span></div>'
        for candle in candles
    )
    return f'<div class="pattern-visual">{candle_html}</div>'

def render_pattern_book():
    st.markdown(
        '<div class="pattern-book-heading">'
        '<div><div class="pattern-book-heading-title">Trading lernen</div>'
        '<div class="pattern-book-heading-subtitle">Candlestick-Muster verstehen und visuell erkennen</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Musterbibliothek öffnen", expanded=False):
        st.caption("Muster sind Hinweise, keine sicheren Vorhersagen. Bestätige sie immer mit Trend, Volumen und Risikomanagement.")
        single_patterns = {"Hammer (bullisch)", "Hanging Man (bärisch)", "Inverted Hammer (bullisch)", "Shooting Star (bärisch)", "Doji (Unentschlossenheit)", "Bullish Marubozu (bullisch)", "Bearish Marubozu (bärisch)", "Spinning Top (Unentschlossenheit)", "Dragonfly Doji (bullisch)", "Gravestone Doji (bärisch)", "Bullish Belt Hold (bullisch)", "Bearish Belt Hold (bärisch)"}
        three_patterns = {"Morning Star (bullisch)", "Evening Star (bärisch)", "3 weiße Soldaten (bullisch)", "3 schwarze Krähen (bärisch)", "Three Inside Up (bullisch)", "Three Inside Down (bärisch)"}
        long_patterns = {"Rising Three Methods (bullisch)", "Falling Three Methods (bärisch)", "Bullish Three Line Strike (bullisch)", "Bearish Three Line Strike (bärisch)"}
        groups = {
            "Einzelkerzen": [pattern for pattern in PATTERNS if pattern in single_patterns],
            "Zwei-Kerzen-Formationen": [pattern for pattern in PATTERNS if pattern not in single_patterns and pattern not in three_patterns and pattern not in long_patterns and "Kicker" not in pattern],
            "Drei-Kerzen-Formationen": [pattern for pattern in PATTERNS if pattern in three_patterns],
            "Mehrkerzen-Formationen": [pattern for pattern in PATTERNS if pattern in long_patterns or "Kicker" in pattern],
        }
        category_tabs = st.tabs([f"{group_name} ({len(patterns)})" for group_name, patterns in groups.items()])
        for category_tab, (group_name, patterns) in zip(category_tabs, groups.items()):
            with category_tab:
                for pattern in patterns:
                    if "bullisch" in pattern:
                        direction, direction_class = "Bullisch", ""
                    elif "bärisch" in pattern:
                        direction, direction_class = "Bärisch", "bearish"
                    else:
                        direction, direction_class = "Neutral", "neutral"
                    st.markdown(
                        f'<div class="pattern-book-card">'
                        f'<div class="pattern-book-title">{pattern}</div>'
                        f'<div class="pattern-book-direction {direction_class}">{direction}</div>'
                        f'{render_pattern_visual(pattern)}'
                        f'<div class="pattern-book-text">{render_pattern_text(pattern)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

render_pattern_book()

def render_market_hours():
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    weekday = now.weekday() < 5
    xetra_open = weekday and time(9, 0) <= now.time() < time(17, 30)
    us_open = weekday and time(15, 30) <= now.time() < time(22, 0)
    xetra_status = "Geöffnet" if xetra_open else "Geschlossen"
    us_status = "Geöffnet" if us_open else "Geschlossen"
    with st.expander("Handelszeiten in deutscher Zeit", expanded=False):
        st.caption("Regelhandel, Montag bis Freitag. Feiertage und Brokerzeiten können abweichen.")
        st.markdown(
            f"**Deutsche Börse / Xetra:** 09:00–17:30 · aktuell: **{xetra_status}**  \n"
            f"**USA / NYSE und Nasdaq:** 15:30–22:00 · aktuell: **{us_status}**  \n"
            "**Krypto:** 24 Stunden, 7 Tage die Woche · Liquidität und Spreads schwanken."
        )
        st.caption("Datenquelle: Yahoo Finance. Aktienkurse können je nach Börse typischerweise verzögert sein; Krypto ist oft näher an Echtzeit, aber nicht garantiert tickgenau. Für echte Echtzeitdaten brauchst du einen lizenzierten Börsen- oder Brokerfeed.")

render_market_hours()

def render_paper_bot():
    with st.expander("Paper-Bot trainieren", expanded=False):
        st.caption("Yahoo-Finance-Daten werden in Trainings- und Testabschnitt geteilt. Es werden nur virtuelle Trades simuliert.")
        bot_options = list(ASSETS.keys()) + st.session_state.watchlist
        bot_asset = st.selectbox("Bot-Asset", bot_options, key="bot_asset")
        bot_interval = st.selectbox("Bot-Intervall", list(INTERVAL_CONFIG.keys()), index=list(INTERVAL_CONFIG.keys()).index("1d"), key="bot_interval")
        bot_capital, bot_risk = st.columns(2)
        with bot_capital:
            bot_initial_capital = st.number_input("Startkapital (€)", min_value=100.0, value=1000.0, step=100.0, key="bot_capital")
        with bot_risk:
            bot_risk_percent = st.number_input("Risiko pro Trade (%)", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="bot_risk")

        auto_optimize = st.checkbox(
            "Parameter automatisch optimieren (Grid-Search, out-of-sample getestet)",
            value=True, key="bot_optimize",
        )
        max_optimization_tests = 243
        if auto_optimize:
            max_optimization_tests = st.slider(
                "Maximale Optimierungs-Tests",
                min_value=10,
                max_value=243,
                value=81,
                step=1,
                help="Mehr Tests prüfen mehr Parameter-Kombinationen, dauern aber länger. 243 prüft alle verfügbaren Kombinationen.",
                key="bot_max_optimization_tests",
            )

        if st.button("Simulation starten", key="run_paper_bot"):
            ticker = ASSETS.get(bot_asset) or bot_asset
            with st.spinner("Historische Yahoo-Finance-Daten werden getestet..."):
                bot_df = load_data(ticker, bot_interval, "Yahoo Finance")
                if bot_df.empty or len(bot_df) < 150:
                    st.error("Für diese Simulation werden mindestens 150 Kerzen benötigt.")
                    st.session_state.paper_bot_result = None
                    st.session_state.paper_bot_best_params = None
                elif auto_optimize:
                    best = optimize_paper_bot(
                        bot_df, bot_initial_capital, bot_risk_percent,
                        max_tests=max_optimization_tests,
                    )
                    if best is None:
                        st.warning("Keine Parameter-Kombination hat genug Trades erzeugt. Versuch ein anderes Intervall/Asset oder deaktiviere die Optimierung.")
                        st.session_state.paper_bot_result = None
                        st.session_state.paper_bot_best_params = None
                    else:
                        st.session_state.paper_bot_result = best["holdout"]
                        st.session_state.paper_bot_best_params = best["params"]
                        st.session_state.paper_bot_test_count = best["tests_run"]
                else:
                    st.session_state.paper_bot_result = simulate_paper_bot(bot_df, bot_initial_capital, bot_risk_percent)
                    st.session_state.paper_bot_best_params = None

        result = st.session_state.paper_bot_result
        if result is not None:
            if st.session_state.get("paper_bot_best_params"):
                bp = st.session_state.paper_bot_best_params
                st.caption(
                    f"Beste gefundene Parameter (out-of-sample getestet): "
                    f"EMA {bp['ema_fast']}/{bp['ema_slow']}, RSI-Periode {bp['rsi_period']}, "
                    f"ATR×{bp['atr_mult']}, Chance/Risiko {bp['reward_risk']}"
                )
                st.caption(f"Geprüfte Parameter-Kombinationen: {st.session_state.get('paper_bot_test_count', '–')}")
            metric_a, metric_b, metric_c, metric_d = st.columns(4)
            metric_a.metric("Endkapital", f'{result["final_equity"]:.2f} €')
            metric_b.metric("Rendite", f'{result["return_percent"]:.1f}%')
            metric_c.metric("Trades", str(result["wins"] + result["losses"]))
            metric_d.metric("Winrate", f'{result["win_rate"]:.1f}%' if result["win_rate"] is not None else "–")
            st.caption(f'{result["wins"]} Gewinne / {result["losses"]} Verluste · Test ausschließlich auf Daten, die der Optimierung nicht bekannt waren.')
            if result["trades"]:
                st.dataframe(pd.DataFrame(result["trades"]).tail(20), use_container_width=True, hide_index=True)

def render_live_paper_trading():
    with st.expander("Live-Trading (Demo / Paper)", expanded=False):
        st.caption("Simulation mit aktuellen Yahoo-Finance-Kerzen. Es werden keine echten Broker-Orders gesendet und kein echtes Geld bewegt.")
        live_asset = st.selectbox("Demo-Asset", list(ASSETS.keys()) + st.session_state.watchlist, key="live_asset")
        live_interval = st.selectbox("Demo-Intervall", list(INTERVAL_CONFIG.keys()), index=list(INTERVAL_CONFIG.keys()).index("1d"), key="live_interval")
        live_capital, live_risk = st.columns(2)
        with live_capital:
            starting_cash = st.number_input("Demo-Startkapital (€)", min_value=100.0, value=1000.0, step=100.0, key="live_capital")
        with live_risk:
            live_risk_percent = st.number_input("Demo-Risiko pro Trade (%)", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="live_risk")

        start_col, check_col = st.columns(2)
        with start_col:
            if st.button("Demo-Konto starten / zurücksetzen", key="start_live_paper"):
                st.session_state.live_paper_account = {
                    "cash": float(starting_cash), "equity": float(starting_cash),
                    "risk_percent": float(live_risk_percent), "position": None,
                    "events": [], "last_candle": None,
                }
        with check_col:
            check_live = st.button("Markt jetzt prüfen", key="check_live_paper")

        account = st.session_state.live_paper_account
        if account is None:
            st.info("Starte zuerst dein Demo-Konto. Danach prüft der Bot beim Klick auf „Markt jetzt prüfen“ ein neues Signal.")
            return

        if check_live:
            ticker = ASSETS.get(live_asset) or live_asset
            live_df = load_data(ticker, live_interval, "Yahoo Finance")
            if live_df.empty or len(live_df) < 30:
                st.warning("Für dieses Asset sind noch nicht genug aktuelle Kerzen verfügbar.")
            else:
                account["risk_percent"] = float(live_risk_percent)
                st.session_state.live_paper_account = run_live_paper_check(live_df, ticker, account)
                account = st.session_state.live_paper_account

        price = account.get("last_setup", {}).get("entry")
        position = account.get("position")
        a, b, c = st.columns(3)
        a.metric("Demo-Kontowert", f'{account["equity"]:.2f} €')
        b.metric("Freies Guthaben", f'{account["cash"]:.2f} €')
        c.metric("Position", "Offen" if position else "Keine")
        if position:
            st.caption(f'Offen: {position["units"]:.6f} Einheiten · Stop {position["stop"]:.4f} · Ziel {position["target"]:.4f}')
        if account.get("last_reason"):
            st.info(account["last_reason"])
        if account["events"]:
            st.markdown("**Käufe, Verkäufe und Begründungen**")
            st.dataframe(pd.DataFrame(account["events"]).iloc[::-1], use_container_width=True, hide_index=True)
        elif price is not None:
            st.caption("Noch keine Order. Der Bot wartet auf ein Long-Signal nach seiner EMA-/RSI-Regel.")

# ------------------------------------------------------------
# Live-Scan über das gesamte Universum: stärkste Long-/Short-Muster
# ------------------------------------------------------------
def batch_load_ohlc(tickers: list[str], interval_key: str) -> dict[str, pd.DataFrame]:
    """Lädt mehrere Ticker in einem yfinance-Aufruf statt einzeln (deutlich schneller)."""
    cfg = INTERVAL_CONFIG[interval_key]
    result: dict[str, pd.DataFrame] = {}
    if not tickers:
        return result
    try:
        raw = yf.download(
            tickers=tickers, period=cfg["period"], interval=cfg["yf_interval"],
            group_by="ticker", threads=True, progress=False,
        )
    except Exception:
        return result
    if raw is None or raw.empty:
        return result
    is_multi = isinstance(raw.columns, pd.MultiIndex)
    for ticker in tickers:
        try:
            if is_multi:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker].copy()
            else:
                df = raw.copy()
            df = df.dropna(how="all")
            if df.empty:
                continue
            if cfg["resample"]:
                df = df.resample(cfg["resample"]).agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum",
                }).dropna()
            df = df.dropna()
            if len(df) < 60:
                continue
            df = df.tail(1000).reset_index()
            date_col = df.columns[0]
            df = df.rename(columns={date_col: "Date"})
            result[ticker] = df
        except Exception:
            continue
    return result

def scan_extreme_patterns(
    universe: dict[str, str], interval_key: str, scan_limit: int,
    bullish_threshold: float = 70.0, bearish_threshold: float = 30.0,
    min_hits: int = 15, chunk_size: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scannt ein Asset-Universum nach Kerzenmustern mit historisch sehr klarer
    Richtung. Bullisch = Muster trat historisch in min_hits Fällen auf und
    führte in >= bullish_threshold % der Fälle zu einem höheren Folgekurs.
    Bärisch entsprechend <= bearish_threshold %. Keine Anlageberatung.
    """
    items = list(universe.items())[:scan_limit]
    label_by_ticker = {ticker: label for label, ticker in items}
    tickers = [ticker for _, ticker in items]

    bullish_rows: list[dict] = []
    bearish_rows: list[dict] = []

    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start:start + chunk_size]
        data_map = batch_load_ohlc(chunk, interval_key)
        for ticker, df in data_map.items():
            try:
                patterns_list = detect_pattern(df)
                last_close = float(df.iloc[-1]["Close"])
                for pattern in patterns_list:
                    if pattern not in PATTERNS:
                        continue
                    probability, hits = historical_probability(df, pattern)
                    if probability is None or hits < min_hits:
                        continue
                    row = {
                        "Asset": label_by_ticker.get(ticker, ticker),
                        "Ticker": ticker,
                        "Muster": pattern,
                        "Trefferquote %": probability,
                        "Vergleichsfälle": hits,
                        "Letzter Kurs": round(last_close, 4),
                    }
                    if probability >= bullish_threshold:
                        bullish_rows.append(row)
                    elif probability <= bearish_threshold:
                        bearish_rows.append(row)
            except Exception:
                continue

    bullish_df = (
        pd.DataFrame(bullish_rows).sort_values(["Trefferquote %", "Vergleichsfälle"], ascending=[False, False])
        if bullish_rows else pd.DataFrame()
    )
    bearish_df = (
        pd.DataFrame(bearish_rows).sort_values(["Trefferquote %", "Vergleichsfälle"], ascending=[True, False])
        if bearish_rows else pd.DataFrame()
    )
    return bullish_df, bearish_df

def _run_and_store_extreme_scan(interval_key: str, limit: int) -> None:
    bullish_df, bearish_df = scan_extreme_patterns(SCANNER_UNIVERSE, interval_key, limit)
    st.session_state.extreme_bullish = bullish_df
    st.session_state.extreme_bearish = bearish_df
    st.session_state.extreme_scanned_at = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y %H:%M:%S")

_HAS_FRAGMENT = hasattr(st, "fragment")
if _HAS_FRAGMENT:
    @st.fragment(run_every=300)
    def _extreme_scan_autorefresh_fragment():
        interval_key = st.session_state.get("extreme_interval", "1d")
        limit = st.session_state.get("extreme_limit", 120)
        with st.spinner(f"Automatischer Scan über {limit} Assets läuft..."):
            _run_and_store_extreme_scan(interval_key, limit)
        st.caption(f"Zuletzt automatisch aktualisiert: {st.session_state.extreme_scanned_at} Uhr (alle 5 Minuten, nur solange dieser Tab offen ist)")

def render_extreme_pattern_scanner():
    with st.expander("Live-Scan: stärkste Long- & Short-Muster (S&P 500 + Top-Kryptos)", expanded=False):
        st.caption(
            f"Durchsucht bis zu {len(SCANNER_UNIVERSE)} der bekanntesten Aktien (alle aktuellen S&P-500-Mitglieder) "
            "und größten Kryptowährungen nach Kerzenmustern mit historisch sehr eindeutiger Richtung. "
            "Kein echter Live-Tick-Feed: Basis sind abgeschlossene Kerzen von Yahoo Finance, die periodisch neu geladen werden."
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            extreme_interval = st.selectbox("Intervall", ["1h", "1d"], index=1, key="extreme_interval")
        with col_b:
            extreme_limit = st.slider(
                "Anzahl gescannter Assets", min_value=20, max_value=len(SCANNER_UNIVERSE),
                value=min(120, len(SCANNER_UNIVERSE)), step=10, key="extreme_limit",
            )
        with col_c:
            auto_refresh = st.checkbox(
                "Alle 5 Min. automatisch aktualisieren (nur bei offenem Tab)",
                value=False, key="extreme_autorefresh",
                disabled=not _HAS_FRAGMENT,
            )
        st.caption(
            f"Ein manueller Scan über {extreme_limit} Assets dauert grob "
            f"{max(1, extreme_limit // 100)}–{max(2, extreme_limit // 40)} Minute(n), abhängig von Yahoo Finance. "
            "Für ständige Überwachung aller ~600 Assets in echter Echtzeit bräuchte es einen bezahlten Marktdaten-Feed "
            "und einen dauerhaft laufenden Server statt einer kostenlosen Streamlit-App."
        )
        if not _HAS_FRAGMENT:
            st.caption("Hinweis: Automatische Aktualisierung benötigt Streamlit ≥ 1.37 (aktuell nicht verfügbar). Bitte manuell scannen.")

        if auto_refresh and _HAS_FRAGMENT:
            _extreme_scan_autorefresh_fragment()
        elif st.button("Jetzt scannen", key="run_extreme_scan"):
            with st.spinner(f"Scanne {extreme_limit} Assets..."):
                _run_and_store_extreme_scan(extreme_interval, extreme_limit)

        scanned_at = st.session_state.get("extreme_scanned_at")
        if scanned_at:
            st.caption(f"Letzter Scan: {scanned_at} Uhr")

        bullish_df = st.session_state.get("extreme_bullish")
        bearish_df = st.session_state.get("extreme_bearish")

        st.markdown("**📈 Stärkste Long-Signale (historisch sehr häufig danach gestiegen)**")
        if bullish_df is not None and not bullish_df.empty:
            st.dataframe(bullish_df.head(15), use_container_width=True, hide_index=True)
        else:
            st.caption("Noch kein Scan oder aktuell kein Asset mit ≥70% historischer Trefferquote bei mind. 15 Vergleichsfällen.")

        st.markdown("**📉 Stärkste Short-Signale (historisch sehr häufig danach gefallen)**")
        if bearish_df is not None and not bearish_df.empty:
            st.dataframe(bearish_df.head(15), use_container_width=True, hide_index=True)
        else:
            st.caption("Noch kein Scan oder aktuell kein Asset mit ≤30% historischer Trefferquote bei mind. 15 Vergleichsfällen.")

        st.caption(
            "Keine Anlageberatung. Eine hohe historische Trefferquote ist keine Garantie für die Zukunft — "
            "Stichprobengröße, veränderte Marktbedingungen, Gebühren und Slippage sind hier nicht eingepreist."
        )

PORTFOLIO_ACCOUNT_KEY = "autonomous_portfolio_v1"

def load_portfolio_account(account_key: str, starting_cash: float, risk_percent: float) -> tuple[dict | None, list[dict]]:
    """Lädt ein Multi-Positionen-Konto. 'position' speichert hier eine Liste offener Positionen."""
    client = get_supabase_client()
    if client is None:
        return None, []
    try:
        result = client.table("scanner_paper_accounts").select("*").eq("account_key", account_key).maybe_single().execute()
        row = result.data
        if not row:
            client.table("scanner_paper_accounts").insert({
                "account_key": account_key, "cash": starting_cash,
                "equity": starting_cash, "risk_percent": risk_percent,
                "position": {"positions": []},
            }).execute()
            row = client.table("scanner_paper_accounts").select("*").eq("account_key", account_key).maybe_single().execute().data
            if not row:
                return None, []
        stored = row.get("position") or {}
        positions = stored.get("positions", []) if isinstance(stored, dict) else []
        account = {
            "cash": float(row["cash"]), "equity": float(row["equity"]),
            "risk_percent": float(row["risk_percent"]), "positions": positions,
            "events": [],
        }
        orders = (
            client.table("scanner_paper_orders").select("created_at,action,ticker,price,units,pnl,reason")
            .eq("account_key", account_key).order("created_at", desc=True).limit(50).execute().data or []
        )
        return account, orders
    except Exception:
        return None, []

def save_portfolio_account(account_key: str, account: dict, events: list[dict]) -> None:
    client = get_supabase_client()
    if client is None:
        return
    client.table("scanner_paper_accounts").update({
        "cash": account["cash"], "equity": account["equity"],
        "risk_percent": account["risk_percent"],
        "position": {"positions": account["positions"]},
        "updated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
    }).eq("account_key", account_key).execute()
    for event in events:
        client.table("scanner_paper_orders").insert({
            "account_key": account_key,
            "action": "BUY" if event["Aktion"] == "KAUF" else "SELL",
            "ticker": event["Asset"], "price": event["Preis"], "units": event["Menge"],
            "pnl": event["Ergebnis"], "reason": event["Warum"],
            "signal": {"source": "autonomous_portfolio_bot"},
        }).execute()

def run_autonomous_portfolio_scan(
    account: dict, interval_key: str, scan_limit: int, max_positions: int,
) -> tuple[dict, pd.DataFrame]:
    """
    Prüft bestehende Positionen auf Stop/Ziel/Trendwechsel und eröffnet neue
    Positionen (bis max_positions) für die besten aktuell erkannten Long-Setups.
    Läuft komplett eigenständig — keine manuelle Auswahl nötig.
    """
    events: list[dict] = []
    positions = account["positions"]
    held_tickers = {p["ticker"] for p in positions}

    # 1) Bestehende Positionen prüfen (Exit-Logik)
    if held_tickers:
        held_data = batch_load_ohlc(list(held_tickers), interval_key)
        still_open = []
        for position in positions:
            ticker = position["ticker"]
            df = held_data.get(ticker)
            if df is None or df.empty:
                still_open.append(position)  # Daten gerade nicht verfügbar: Position unangetastet lassen
                continue
            setup = calculate_trade_setup(df)
            price = float(setup["entry"])
            close_reason = None
            if price <= position["stop"]:
                close_reason = "Stop-Loss erreicht"
            elif price >= position["target"]:
                close_reason = "Take-Profit erreicht"
            elif setup["direction"] == "short":
                close_reason = "Trendwechsel: EMA/RSI geben ein Verkaufssignal"
            if close_reason:
                proceeds = position["units"] * price
                account["cash"] += proceeds
                pnl = proceeds - position["cost"]
                events.append({
                    "Zeit": str(df.iloc[-1]["Date"]), "Aktion": "VERKAUF", "Asset": ticker,
                    "Preis": round(price, 4), "Menge": round(position["units"], 6),
                    "Ergebnis": round(pnl, 2), "Warum": close_reason,
                })
            else:
                still_open.append(position)
        positions = still_open
        held_tickers = {p["ticker"] for p in positions}

    # 2) Neue Kandidaten suchen, solange Plätze frei sind
    candidates_rows: list[dict] = []
    free_slots = max_positions - len(positions)
    if free_slots > 0 and account["cash"] > 1.0:
        universe_items = [
            (label, ticker) for label, ticker in list(SCANNER_UNIVERSE.items())[:scan_limit]
            if ticker not in held_tickers
        ]
        scan_tickers = [ticker for _, ticker in universe_items]
        label_by_ticker = {ticker: label for label, ticker in universe_items}
        data_map = batch_load_ohlc(scan_tickers, interval_key)
        candidates = []
        for ticker, df in data_map.items():
            try:
                setup = calculate_trade_setup(df)
                if not setup["signal"].startswith("KAUFEN") or setup["stop"] is None:
                    continue
                volume_score = min(float(setup.get("volume_ratio") or 0), 3.0)
                rsi = float(setup.get("rsi") or 0)
                score = volume_score * 10 + (70 - abs(60 - rsi))
                candidates.append({
                    "Asset": label_by_ticker.get(ticker, ticker), "Ticker": ticker,
                    "Score": round(score, 1), "RSI": round(rsi, 1), "_df": df, "_setup": setup,
                })
            except Exception:
                continue
        candidates.sort(key=lambda item: item["Score"], reverse=True)
        candidates_rows = [{k: v for k, v in c.items() if not k.startswith("_")} for c in candidates]

        for candidate in candidates:
            if free_slots <= 0 or account["cash"] <= 1.0:
                break
            setup = candidate["_setup"]
            df = candidate["_df"]
            price = float(setup["entry"])
            risk_per_unit = max(price - float(setup["stop"]), 1e-9)
            risk_budget = account["equity"] * account["risk_percent"] / 100
            units = min(risk_budget / risk_per_unit, account["cash"] / price)
            if units <= 0:
                continue
            cost = units * price
            account["cash"] -= cost
            positions.append({
                "ticker": candidate["Ticker"], "units": units, "cost": cost,
                "stop": float(setup["stop"]), "target": float(setup["target"]),
                "entry_price": price, "opened_at": str(df.iloc[-1]["Date"]),
            })
            events.append({
                "Zeit": str(df.iloc[-1]["Date"]), "Aktion": "KAUF", "Asset": candidate["Ticker"],
                "Preis": round(price, 4), "Menge": round(units, 6), "Ergebnis": 0.0,
                "Warum": f"Bestes verfügbares Long-Setup (Score {candidate['Score']}) unter freien Portfolio-Plätzen",
            })
            free_slots -= 1

    account["positions"] = positions
    position_value = sum(p["units"] * p.get("_last_price", p["cost"] / p["units"]) for p in positions) if positions else 0.0
    account["equity"] = account["cash"] + position_value
    account["events"] = events
    return account, pd.DataFrame(candidates_rows)

def render_autonomous_portfolio_bot():
    with st.expander("Autonomer Portfolio-Bot · handelt selbstständig über mehrere Assets", expanded=False):
        st.caption(
            "Dieser Bot verwaltet ein gemeinsames, in Supabase gespeichertes Demo-Portfolio ohne echtes Geld. "
            "Er prüft bestehende Positionen auf Stop-Loss/Take-Profit/Trendwechsel und eröffnet eigenständig neue "
            "Long-Positionen, wenn ein Asset im Scan-Universum ein starkes Setup zeigt — bis zur eingestellten "
            "Anzahl gleichzeitiger Positionen. Keine echten Orders, keine Anlageberatung."
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            portfolio_interval = st.selectbox("Intervall", ["1h", "1d"], index=1, key="portfolio_interval")
        with col_b:
            portfolio_scan_limit = st.slider(
                "Scan-Umfang (Assets)", min_value=20, max_value=len(SCANNER_UNIVERSE),
                value=min(150, len(SCANNER_UNIVERSE)), step=10, key="portfolio_scan_limit",
            )
        with col_c:
            portfolio_max_positions = st.slider("Max. gleichzeitige Positionen", min_value=1, max_value=15, value=5, key="portfolio_max_positions")

        cap_col, risk_col = st.columns(2)
        with cap_col:
            portfolio_cash = st.number_input("Startkapital (€)", min_value=100.0, value=10000.0, step=100.0, key="portfolio_cash")
        with risk_col:
            portfolio_risk = st.number_input("Risiko pro neuer Position (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="portfolio_risk")

        auto_refresh = st.checkbox(
            "Alle 5 Min. automatisch prüfen & selbstständig handeln (nur solange Tab offen)",
            value=False, key="portfolio_autorefresh", disabled=not _HAS_FRAGMENT,
        )
        if not _HAS_FRAGMENT:
            st.caption("Automatische Ausführung benötigt Streamlit ≥ 1.37. Bitte manuell auf 'Jetzt prüfen' klicken.")
        st.caption(
            "Wichtig: Auch mit aktivierter Automatik läuft das nur, solange dieser Browser-Tab offen ist — "
            "kein echter 24/7-Serverdienst. Für dauerhafte Überwachung bräuchte es einen extern gehosteten Dienst "
            "oder einen kostenlosen Pinger, der die App wach hält, plus einen eigenen Cron-Trigger."
        )

        def _check_and_trade():
            account, orders_before = load_portfolio_account(PORTFOLIO_ACCOUNT_KEY, float(portfolio_cash), float(portfolio_risk))
            if account is None:
                st.warning("Der Portfolio-Bot braucht die Supabase-Secrets und die Scanner-Tabellen (siehe oben im Markt-Scanner).")
                error_detail = st.session_state.get("_supabase_last_error")
                if error_detail:
                    st.code(error_detail, language="text")
                return
            account["risk_percent"] = float(portfolio_risk)
            account, candidates_df = run_autonomous_portfolio_scan(
                account, portfolio_interval, portfolio_scan_limit, portfolio_max_positions,
            )
            save_portfolio_account(PORTFOLIO_ACCOUNT_KEY, account, account["events"])
            st.session_state.portfolio_last_candidates = candidates_df
            st.session_state.portfolio_last_run = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y %H:%M:%S")

        if _HAS_FRAGMENT and auto_refresh:
            @st.fragment(run_every=300)
            def _portfolio_autorefresh_fragment():
                with st.spinner("Prüfe Portfolio und scanne nach neuen Setups..."):
                    _check_and_trade()
                st.caption(f"Zuletzt automatisch geprüft: {st.session_state.get('portfolio_last_run', '–')} Uhr")
            _portfolio_autorefresh_fragment()
        elif st.button("Jetzt prüfen & ggf. handeln", key="run_portfolio_bot"):
            with st.spinner("Prüfe Portfolio und scanne nach neuen Setups..."):
                _check_and_trade()

        account, orders = load_portfolio_account(PORTFOLIO_ACCOUNT_KEY, float(portfolio_cash), float(portfolio_risk))
        if account is None:
            st.warning("Der Portfolio-Bot braucht die Supabase-Secrets und die Scanner-Tabellen (siehe oben im Markt-Scanner).")
            error_detail = st.session_state.get("_supabase_last_error")
            if error_detail:
                st.code(error_detail, language="text")
            return

        last_run = st.session_state.get("portfolio_last_run")
        if last_run:
            st.caption(f"Letzte Prüfung: {last_run} Uhr")

        positions = account["positions"]
        held_tickers = [p["ticker"] for p in positions]
        live_prices: dict[str, float] = {}
        if held_tickers:
            live_data = batch_load_ohlc(held_tickers, portfolio_interval)
            for ticker, df in live_data.items():
                if not df.empty:
                    live_prices[ticker] = float(df.iloc[-1]["Close"])

        position_value = sum(p["units"] * live_prices.get(p["ticker"], p["cost"] / p["units"]) for p in positions)
        equity = account["cash"] + position_value

        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Gesamtwert (Cash + Positionen)", f'{equity:.2f} €')
        metric_b.metric("Freies Guthaben", f'{account["cash"]:.2f} €')
        metric_c.metric("Offene Positionen", f'{len(positions)} / {portfolio_max_positions}')

        st.markdown("**Offene Positionen — Live-Übersicht**")
        if positions:
            overview_rows = []
            for p in positions:
                current_price = live_prices.get(p["ticker"], p["cost"] / p["units"])
                entry_price = p.get("entry_price", p["cost"] / p["units"])
                unrealized = (current_price - entry_price) * p["units"]
                unrealized_pct = ((current_price / entry_price) - 1) * 100 if entry_price else 0.0
                overview_rows.append({
                    "Ticker": p["ticker"], "Einstieg": round(entry_price, 4),
                    "Aktueller Kurs": round(current_price, 4), "Stück": round(p["units"], 6),
                    "Stop-Loss": round(p["stop"], 4), "Take-Profit": round(p["target"], 4),
                    "Unreal. Ergebnis (€)": round(unrealized, 2), "Unreal. Ergebnis (%)": round(unrealized_pct, 2),
                    "Eröffnet": p.get("opened_at", "–"),
                })
            st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Aktuell keine offenen Positionen. Der Bot eröffnet automatisch, sobald ein starkes Long-Setup im Scan-Universum auftaucht.")

        candidates_df = st.session_state.get("portfolio_last_candidates")
        if candidates_df is not None and not candidates_df.empty:
            st.markdown("**Zuletzt beste gefundene Kauf-Kandidaten (nicht zwingend gekauft, falls Plätze/Kapital fehlten)**")
            st.dataframe(candidates_df.head(10), use_container_width=True, hide_index=True)

        if orders:
            st.markdown("**Handelshistorie des Portfolio-Bots**")
            display_orders = pd.DataFrame(orders).rename(columns={
                "created_at": "Zeit", "action": "Aktion", "ticker": "Asset",
                "price": "Preis", "units": "Menge", "pnl": "Ergebnis", "reason": "Warum",
            })
            st.dataframe(display_orders, use_container_width=True, hide_index=True)

        st.caption(
            "Keine Anlageberatung. Simulierter Handel ohne Gebühren, Slippage oder Orderausführungsrisiko — "
            "reale Ergebnisse würden davon abweichen."
        )

def render_market_scanner_paper_bot():
    with st.expander("Markt-Scanner · gemeinsamer Demo-Bot", expanded=False):
        st.caption("Ein gemeinsames, in Supabase gespeichertes Paper-Konto. Der Scanner bewertet liquide Aktien, ETFs und Krypto und eröffnet höchstens eine Long-Position. Keine echten Orders.")
        scanner_interval = st.selectbox("Scanner-Intervall", ["1h", "1d"], index=1, key="scanner_interval")
        scanner_limit = st.slider("Assets pro Scan", min_value=3, max_value=len(SCANNER_UNIVERSE), value=10, key="scanner_limit")
        scanner_cash, scanner_risk = st.columns(2)
        with scanner_cash:
            scanner_starting_cash = st.number_input("Startkapital für gemeinsamen Bot (€)", min_value=100.0, value=10000.0, step=100.0, key="scanner_cash")
        with scanner_risk:
            scanner_risk_percent = st.number_input("Risiko je Scanner-Trade (%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="scanner_risk")
        account, orders = load_scanner_account(float(scanner_starting_cash), float(scanner_risk_percent))
        if account is None:
            st.warning("Der gemeinsame Scanner braucht die Supabase-Secrets und die neuen Scanner-Tabellen. Führe zuerst die aktuelle SQL-Datei aus.")
            error_detail = st.session_state.get("_supabase_last_error")
            if error_detail:
                st.code(error_detail, language="text")
            return
        if st.button("Markt scannen & Demo-Bot prüfen", key="run_scanner"):
            with st.spinner("Scanne Markt und prüfe das beste Setup..."):
                account["risk_percent"] = float(scanner_risk_percent)
                account, candidates = scan_and_trade_paper_market(account, scanner_interval, scanner_limit)
                st.session_state.scanner_candidates = candidates
                _, orders = load_scanner_account(float(scanner_starting_cash), float(scanner_risk_percent))
        else:
            candidates = st.session_state.get("scanner_candidates")

        position = account.get("position")
        a, b, c = st.columns(3)
        a.metric("Gemeinsamer Kontowert", f'{account["equity"]:.2f} €')
        b.metric("Freies Guthaben", f'{account["cash"]:.2f} €')
        c.metric("Position", position.get("ticker") if position else "Keine")
        if account.get("last_reason"):
            st.info(account["last_reason"])
        if candidates is not None and not candidates.empty:
            st.markdown("**Beste Scanner-Signale**")
            st.dataframe(candidates.head(10), use_container_width=True, hide_index=True)
        if orders:
            st.markdown("**Dauerhafte Demo-Order-Historie**")
            display_orders = pd.DataFrame(orders).rename(columns={"created_at": "Zeit", "action": "Aktion", "ticker": "Asset", "price": "Preis", "units": "Menge", "pnl": "Ergebnis", "reason": "Warum"})
            st.dataframe(display_orders, use_container_width=True, hide_index=True)

        st.markdown("**Chart ansehen**")
        chart_label = st.selectbox("Scanner-Chart-Asset", list(SCANNER_UNIVERSE.keys()), key="scanner_chart_asset")
        chart_ticker = SCANNER_UNIVERSE[chart_label]
        try:
            chart_data = load_data(chart_ticker, scanner_interval, "Yahoo Finance")
            if not chart_data.empty:
                render_candlestick_chart(chart_data, "Scanner-Chart", chart_ticker)
            else:
                st.warning("Für dieses Chart sind gerade keine Marktdaten verfügbar.")
        except Exception:
            st.warning("Chart konnte gerade nicht geladen werden.")

render_market_scanner_paper_bot()
render_autonomous_portfolio_bot()
render_extreme_pattern_scanner()
render_ml_predictor()

# ------------------------------------------------------------
# Tabs: Einzelanalyse vs. Meine Positionen
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["Einzelanalyse", "Meine Positionen"])

# ============================================================
# TAB 1 – Einzelanalyse
# ============================================================
with tab1:
    st.markdown('<div class="section-label">Asset</div>', unsafe_allow_html=True)
    combined_options = list(ASSETS.keys()) + st.session_state.watchlist
    asset_choice = st.selectbox("Asset wählen", combined_options, label_visibility="collapsed", key="single_asset")

    st.markdown('<div class="section-label">Zeitrahmen (Kerzen-Intervall)</div>', unsafe_allow_html=True)
    interval_label = st.selectbox(
        "Zeitrahmen",
        list(INTERVAL_CONFIG.keys()),
        index=list(INTERVAL_CONFIG.keys()).index("1d"),
        label_visibility="collapsed",
        key="single_interval",
    )

    button_left, button_center, button_right = st.columns([1, 2, 1])
    with button_center:
        run = st.button("Chart analysieren", key="single_run", use_container_width=True)

    if run:
        ticker = ASSETS.get(asset_choice) or asset_choice

        with st.spinner("Lade Kursdaten..."):
            try:
                st.session_state.single_result = analyze_ticker(ticker, interval_label, data_source)
            except Exception as error:
                st.session_state.single_result = None
                st.error(str(error))
            st.session_state.single_ai_text = None
            st.session_state.single_ai_key = None

        if st.session_state.single_result is None:
            st.error("Nicht genügend Kursdaten gefunden. Bitte anderes Asset/Intervall wählen.")

    result = st.session_state.single_result
    if result is not None:
        render_result_card(
            result["ticker"], result["pattern"], result["probability"],
            result["hits"], len(result["df"]), interval_label,
        )
        render_trade_setup(result["trade_setup"], result["ticker"], interval_label)
        render_candlestick_chart(result["df"], result["pattern"], result["ticker"])

        provider_key = {
            "Gemini": st.session_state.gemini_api_key,
            "OpenAI": st.session_state.openai_api_key,
            "Claude": st.session_state.anthropic_api_key,
        }.get(ai_provider, "")
        if ai_provider != "Keiner" and provider_key:
            st.caption("Die KI ist optional und wird nur nach Klick auf den folgenden Button angefragt.")
            if st.button("KI-Einschätzung laden", key=f"single_ai_{result['ticker']}"):
                with st.spinner(f"{ai_provider} erstellt eine kurze Einschätzung..."):
                    st.session_state.single_ai_text = get_ai_analysis(
                        ai_provider, provider_key, result["ticker"], result["pattern"],
                        result["probability"], result["df"],
                    )
            if st.session_state.single_ai_text:
                st.markdown(
                    f'<div class="ai-card"><div class="ai-label">KI-Einschätzung</div>{st.session_state.single_ai_text.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Wähle oben einen KI-Anbieter und hinterlege den passenden API-Key für eine optionale Einschätzung.")

        st.markdown(
            '<div class="disclaimer">Keine Anlageberatung. Rein statistische/historische '
            'Auswertung, keine Garantie für zukünftige Kursbewegungen.</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# TAB 2 – Meine Positionen (Watchlist + CSV-Import)
# ============================================================
with tab2:
    st.markdown(
        """
        <div class="info-card">
        Trade Republic bietet keine offizielle Schnittstelle für Drittanbieter-Logins –
        ein direkter Login mit deinem TR-Passwort in einer fremden App wäre nicht sicher.
        Trag deine Positionen stattdessen hier ein (manuell oder per CSV),
        die Kurse holt sich die App automatisch über Yahoo Finance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Ticker manuell hinzufügen</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        new_ticker = st.text_input(
            "Ticker", label_visibility="collapsed",
            placeholder="z.B. SAP.DE, AAPL, MSFT, BTC-USD",
            key="new_ticker_input",
        )
    with col_b:
        add_clicked = st.button("Add", key="add_ticker_btn")

    if add_clicked and new_ticker.strip():
        t = new_ticker.strip().upper()
        if t not in st.session_state.watchlist:
            st.session_state.watchlist.append(t)
        st.rerun()

    st.caption("Tipp: Deutsche Aktien meist mit **.DE** (z.B. SAP.DE), US-Aktien ohne Zusatz (z.B. AAPL).")

    if st.session_state.watchlist:
        st.markdown('<div class="section-label">Deine Watchlist</div>', unsafe_allow_html=True)
        chips_html = "".join(f'<span class="watch-chip">{t}</span>' for t in st.session_state.watchlist)
        st.markdown(chips_html, unsafe_allow_html=True)

        remove_choice = st.multiselect(
            "Ticker entfernen", st.session_state.watchlist,
            placeholder="Ticker zum Entfernen auswählen", key="remove_select",
        )
        if remove_choice and st.button("Entfernen", key="remove_btn"):
            st.session_state.watchlist = [t for t in st.session_state.watchlist if t not in remove_choice]
            st.rerun()

    st.markdown('<div class="section-label">Positionen per CSV importieren</div>', unsafe_allow_html=True)
    st.caption(
        "Lade eine CSV-Datei mit deinen Positionen hoch (z.B. selbst exportiert oder "
        "abgetippt). Erwartet wird mindestens eine Spalte mit dem Ticker-Symbol."
    )
    csv_file = st.file_uploader("CSV-Datei", type=["csv"], label_visibility="collapsed")

    if csv_file is not None:
        try:
            csv_df = pd.read_csv(csv_file)
        except Exception:
            csv_file.seek(0)
            csv_df = pd.read_csv(io.StringIO(csv_file.getvalue().decode("utf-8", errors="ignore")), sep=";")

        st.dataframe(csv_df.head(20), use_container_width=True, height=180)

        ticker_col = None
        for candidate in ["Ticker", "ticker", "Symbol", "symbol", "TICKER"]:
            if candidate in csv_df.columns:
                ticker_col = candidate
                break

        if ticker_col is None:
            ticker_col = st.selectbox(
                "Welche Spalte enthält den Ticker?", csv_df.columns.tolist(), key="csv_ticker_col",
            )
        else:
            st.caption(f"Ticker-Spalte automatisch erkannt: **{ticker_col}**")

        if st.button("Aus CSV in Watchlist übernehmen", key="import_csv_btn"):
            new_tickers = (
                csv_df[ticker_col].dropna().astype(str).str.strip().str.upper().unique().tolist()
            )
            added = 0
            for t in new_tickers:
                if t and t not in st.session_state.watchlist:
                    st.session_state.watchlist.append(t)
                    added += 1
            st.success(f"{added} neue Ticker zur Watchlist hinzugefügt.")
            st.rerun()

    if st.session_state.watchlist:
        st.markdown('<div class="section-label">Alle Positionen analysieren</div>', unsafe_allow_html=True)
        batch_interval = st.selectbox(
            "Zeitrahmen für alle",
            list(INTERVAL_CONFIG.keys()),
            index=list(INTERVAL_CONFIG.keys()).index("1d"),
            key="batch_interval",
        )

        st.markdown('<div class="cta-btn">', unsafe_allow_html=True)
        analyze_all = st.button("Alle analysieren", key="analyze_all_btn")
        st.markdown('</div>', unsafe_allow_html=True)

        batch_provider_key = {
            "Gemini": st.session_state.gemini_api_key,
            "OpenAI": st.session_state.openai_api_key,
            "Claude": st.session_state.anthropic_api_key,
        }.get(ai_provider, "")
        use_ai_batch = st.checkbox(
            "KI-Einschätzungen für alle Ticker anfordern",
            value=False,
            disabled=not bool(batch_provider_key) or ai_provider == "Keiner",
            key="use_ai_batch",
            help="Verbraucht eine Gemini-Anfrage pro erfolgreich geladenem Ticker.",
        )
        if not batch_provider_key or ai_provider == "Keiner":
            st.caption("Wähle oben einen KI-Anbieter und hinterlege den passenden API-Key für KI-Einschätzungen.")

        if analyze_all:
            progress = st.progress(0.0, text="Starte Analyse...")
            results = []
            for i, t in enumerate(st.session_state.watchlist):
                progress.progress((i + 1) / len(st.session_state.watchlist), text=f"Analysiere {t}...")
                try:
                    res = analyze_ticker(t, batch_interval, data_source)
                except Exception as error:
                    st.warning(f"{t}: {error}")
                    res = None
                if res:
                    if batch_provider_key and ai_provider != "Keiner" and use_ai_batch:
                        res["ai_text"] = get_ai_analysis(
                            ai_provider, batch_provider_key, t, res["pattern"], res["probability"], res["df"]
                        )
                    else:
                        res["ai_text"] = None
                    results.append(res)
            progress.empty()

            if not results:
                st.error("Für keinen deiner Ticker konnten Daten geladen werden. Bitte Symbole prüfen.")
            else:
                for res in sorted(
                    results,
                    key=lambda r: r["probability"] if r["probability"] is not None else -1,
                    reverse=True,
                ):
                    render_mini_card(res["ticker"], res["pattern"], res["probability"])
                    render_candlestick_chart(res["df"], res["pattern"], res["ticker"], n_candles=25)
                    if res["ai_text"]:
                        ai_html = res["ai_text"].replace(chr(10), "<br>")
                        st.markdown(
                            f'<div class="ai-card" style="margin-top:-0.3em;">'
                            f'<div class="ai-label">{res["ticker"]}</div>{ai_html}</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    '<div class="disclaimer">Keine Anlageberatung. Rein statistische/historische '
                    'Auswertung, keine Garantie für zukünftige Kursbewegungen.</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.caption("Noch keine Positionen in der Watchlist.")
