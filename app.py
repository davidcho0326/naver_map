from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import os
import logging
import sys
import traceback
import json
from openai import OpenAI
from dotenv import load_dotenv
from rag_system import RAGSystem
from sqlalchemy import text
from settings import create_postgres_engine_by_sqlalchemy
from difflib import SequenceMatcher
from sqlalchemy.exc import SQLAlchemyError

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# OpenAI 관련 로그 레벨 설정
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 환경 변수 로드
load_dotenv()

try:
    app = Flask(__name__, static_folder=None)
    CORS(app, resources={r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }})

    NAVER_CLIENT_ID = os.getenv('NAVER_CLIENT_ID')
    NAVER_CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET')

    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        logger.error("네이버 API 키가 설정되지 않았습니다.")
        raise ValueError("네이버 API 키가 필요합니다.")

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # RAG 시스템 초기화 로깅 추가
    logger.debug("RAG 시스템 초기화 시작")
    rag = RAGSystem()
    logger.debug("RAG 시스템 초기화 완료")

    # 한국의 주요 지역 정보 - GPT 컨텍스트 강화용
    KOREA_LOCATIONS = """
    한국의 주요 지역 정보:
    - 서울: 강남구(압구정동, 청담동, 삼성동, 역삼동, 논현동), 서초구(서초동, 반포동, 방배동), 
           종로구(인사동, 삼청동, 북촌), 중구(명동, 을지로, 동대문), 마포구(홍대, 연남동, 합정동), 
           용산구(이태원, 한남동), 강서구, 강동구, 노원구 등
    - 부산: 해운대구(해운대, 마린시티), 수영구(광안리), 남구(용호동), 중구(남포동, 광복동), 서구(송도) 등
    - 인천: 중구(차이나타운, 월미도), 연수구(송도), 남동구(구월동) 등
    - 대구: 중구(동성로), 수성구(범어동, 두산동), 달서구(상인동) 등
    - 대전: 중구(은행동), 서구(둔산동), 유성구(봉명동) 등
    - 광주: 동구(충장로), 서구(상무지구), 북구(용봉동) 등
    - 제주: 제주시(이도동, 연동), 서귀포시(중문, 성산) 등
    """

    # 장소 유형 정보 - GPT 컨텍스트 강화용
    PLACE_TYPES = """
    장소 유형별 키워드:
    - 음식점/맛집: 식당, 레스토랑, 맛집, 음식점, 한식, 중식, 일식, 양식, 분식, 패스트푸드, 뷔페, 베이커리, 디저트
    - 카페: 카페, 커피숍, 디저트, 브런치, 베이커리, 차, 음료
    - 쇼핑: 백화점, 쇼핑몰, 마트, 상점, 시장, 아울렛, 편의점, 가게
    - 의료: 병원, 의원, 약국, 치과, 한의원, 보건소, 의료원, 클리닉
    - 교통: 지하철역, 버스정류장, 터미널, 공항, 기차역, 정류장, 역
    - 숙박: 호텔, 모텔, 게스트하우스, 리조트, 펜션, 숙소, 콘도
    - 관광: 명소, 관광지, 박물관, 미술관, 전시관, 유적지, 공원, 테마파크
    - 문화/예술: 영화관, 공연장, 전시장, 극장, 문화센터, 스튜디오
    - 교육: 학교, 학원, 도서관, 독서실, 교육원, 대학교, 유치원
    - 스포츠/레저: 체육관, 수영장, 헬스장, 운동장, 골프장, 스키장
    """

    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # 대화 기록을 저장할 전역 딕셔너리 추가
    conversation_history = {}

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/search', methods=['POST'])
    def search():
        try:
            if not request.is_json:
                logger.error("JSON이 아닌 요청이 들어왔습니다.")
                return jsonify({'error': '요청이 JSON 형식이어야 합니다.'}), 400

            data = request.get_json()
            query = data.get('query', '')
            
            if not query:
                logger.error("검색어가 비어있습니다.")
                return jsonify({'error': '검색어를 입력해주세요.', 'type': 'error'}), 400
            
            # 시설 유형 분석
            facility_type, confidence_score, matched_keyword = analyze_facility_type(query)
            
            if not facility_type:
                logger.warning("지원되지 않는 시설 유형")
                return jsonify({
                    'error': '검색 결과를 찾을 수 없습니다. 병원, 음식점, 카페, 약국 중 하나를 선택해 주세요.',
                    'type': 'error'
                }), 404
            
            if confidence_score < 0.3:
                logger.warning(f"낮은 신뢰도 점수: {confidence_score}")
                return jsonify({
                    'error': '시설 유형을 정확히 파악할 수 없습니다. 더 구체적으로 말씀해 주세요.',
                    'type': 'error'
                }), 400

            try:
                logger.info("\n[RAG 검색 시작]")
                logger.info(f"- 시설 유형: {facility_type} (신뢰도: {confidence_score}, 키워드: {matched_keyword})")
                logger.info(f"검색 파라미터 - 쿼리: '{query}', 시설 유형: '{facility_type}', top_k: 3")
                
                # RAG 시스템으로 시설 검색
                search_results = rag.search_similar_documents(query, facility_type, top_k=3)
                
                if search_results:
                    logger.info(f"검색 결과 {len(search_results)}개 찾음")
                    logger.debug(f"검색 결과 상세: {json.dumps(search_results, ensure_ascii=False, indent=2)}")
                    
                    # GPT로 자연어 응답 생성
                    response_prompt = f"""
                    F&F 신사옥 주변의 {facility_type} 검색 결과입니다:
                    {json.dumps(search_results, ensure_ascii=False, indent=2)}
                    
                    이 정보를 자연스러운 한국어로 설명해주세요. 각 시설의 이름, 위치(지하철역 기준), 영업시간, 특징적인 정보를 포함해주세요.
                    응답은 친근하고 대화체로 작성해주세요.
                    """
                    
                    natural_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "당신은 친근한 말투로 주변 시설을 추천해주는 도우미입니다."},
                            {"role": "user", "content": response_prompt}
                        ]
                    )
                    
                    response_text = natural_response.choices[0].message.content
                    logger.info(f"GPT 응답: {response_text}")
                    
                    # 검색 결과를 지도에 표시하기 위한 좌표 정보 추가
                    for place in search_results:
                        try:
                            # 네이버 지도 API로 좌표 검색
                            url = "https://openapi.naver.com/v1/search/local.json"
                            headers = {
                                "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
                                "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET
                            }
                            params = {
                                "query": f"{place['name']} {place['address']}",
                                "display": 1
                            }
                            
                            map_response = requests.get(url, headers=headers, params=params)
                            map_data = map_response.json()
                            
                            if map_data.get('items'):
                                item = map_data['items'][0]
                                place['x'] = float(item['mapx'])  # 경도
                                place['y'] = float(item['mapy'])  # 위도
                        except Exception as e:
                            logger.error(f"좌표 검색 실패: {str(e)}")
                            continue
                    
                    response_data = {
                        'type': 'places',
                        'places': search_results,
                        'response': response_text
                    }
                    logger.info("\n[응답 전송]")
                    logger.info(f"응답 데이터: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
                    return jsonify(response_data)
                    
                else:
                    logger.warning("검색 결과가 없습니다.")
                    return jsonify({
                        'error': f'주변에서 {facility_type}을(를) 찾을 수 없습니다.',
                        'type': 'error'
                    }), 404
                
            except Exception as e:
                logger.error(f"\n[오류 발생]\n{str(e)}\n{traceback.format_exc()}")
                return jsonify({
                    'error': '시설 검색 중 오류가 발생했습니다.',
                    'type': 'error'
                }), 500

        except Exception as e:
            logger.error(f"\n[오류 발생]\n{str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500

    @app.route('/api/search', methods=['POST', 'GET'])
    def search_api():
        if request.method == 'GET':
            query = request.args.get('query', '')
        else:  # POST
            data = request.get_json()
            query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
        
        logger.debug(f"검색 요청: {query}")
        
        try:
            # 사용자 위치 정보 가져오기
            user_location = None
            if request.method == 'POST' and request.get_json():
                user_location = request.get_json().get('user_location')
            
            # 쿼리 분석
            analysis = analyze_query(query, user_location)
            logger.debug(f"쿼리 분석 결과: {analysis}")
            if analysis['query_type'] == 'directions':
                logger.info("\n[길찾기 검색 시작]")
                target_name = analysis['target_location']
                logger.info(f"- 목적지: {target_name}")
                
                try:
                    # FAISS에서 병원 정보 검색
                    search_results = rag.search_similar_documents(target_name, "병원", top_k=5)  # top_k를 5로 증가
                    
                    if not search_results:
                        return jsonify({
                            'type': 'chat',
                            'response': f"죄송합니다. '{target_name}'의 정보를 찾을 수 없습니다."
                        })
                    
                    # 검색 결과 중에서 가장 유사한 병원 찾기
                    best_match = None
                    highest_similarity = 0
                    
                    for result in search_results:
                        # 전체 이름과의 유사도 계산
                        full_similarity = similar(target_name, result['name'])
                        
                        # 부분 문자열 포함 여부 확인 (공백 제거 후 비교)
                        target_clean = target_name.replace(" ", "").lower()
                        result_clean = result['name'].replace(" ", "").lower()
                        contains_partial = target_clean in result_clean
                        
                        # 입력된 단어들과의 개별 유사도 계산
                        target_words = target_name.split()
                        name_words = result['name'].split()
                        word_similarities = []
                        for tw in target_words:
                            tw_clean = tw.lower()
                            best_word_match = max([similar(tw_clean, nw.lower()) for nw in name_words])
                            word_similarities.append(best_word_match)
                        avg_word_similarity = sum(word_similarities) / len(word_similarities)
                        
                        # 종합 유사도 점수 계산 (가중치 조정)
                        similarity_score = max(
                            full_similarity * 0.3,      # 전체 문자열 유사도 (30%)
                            avg_word_similarity * 0.3,  # 단어별 평균 유사도 (30%)
                            0.9 if contains_partial else 0  # 부분 문자열 포함 보너스 (90%)
                        )
                        
                        logger.debug(f"""
                        유사도 분석:
                        - 검색어: {target_name}
                        - 병원명: {result['name']}
                        - 전체 유사도: {full_similarity}
                        - 단어별 평균 유사도: {avg_word_similarity}
                        - 부분 문자열 포함: {contains_partial}
                        - 최종 유사도 점수: {similarity_score}
                        """)
                        
                        if similarity_score > highest_similarity:
                            highest_similarity = similarity_score
                            best_match = result
                    
                    if highest_similarity >= 0.2:  # 유사도 임계값 낮춤 (20%)
                        target_place = best_match
                        target_name = target_place['name']  # 정확한 병원 이름으로 업데이트
                        target_address = target_place['address']
                        
                        logger.info(f"- 검색된 병원: {target_name}")
                        logger.info(f"- 검색된 주소: {target_address}")
                        logger.info(f"- 유사도 점수: {highest_similarity}")
                    
                    # 주소를 좌표로 변환
                    location_result = geocode_location(target_address)
                    
                    if not location_result.get('addresses'):
                        return jsonify({
                            'type': 'chat',
                            'response': f"죄송합니다. '{target_name}'의 주소를 찾을 수 없습니다."
                        })
                    
                    target_coords = location_result['addresses'][0]
                    
                    # 출발지 좌표 (F&F 신사옥)
                    start_coords = {
                        'x': '127.0310195',  # 경도
                        'y': '37.4982517'    # 위도
                    }
                    
                    # 길찾기 API 호출
                    route_result = get_directions_data(
                        f"{start_coords['x']},{start_coords['y']}",
                        f"{target_coords['x']},{target_coords['y']}"
                    )
                    
                    if route_result.get('route'):
                        summary = route_result['route']['trafast'][0]['summary']
                        distance = summary['distance']
                        duration_minutes = int(summary['duration'] / (1000 * 60))
                        
                        logger.debug(f"""
                        [길찾기 결과]
                        - 목적지 이름: {target_name}
                        - 목적지 주소: {target_address}
                        - 목적지 좌표: {target_coords['x']}, {target_coords['y']}
                        - 거리: {distance}m
                        - 시간: {duration_minutes}분
                        """)
                        
                        # GPT로 길찾기 결과 자연어 변환
                        directions_prompt = f"""
                        F&F 신사옥에서 {target_name}까지의 길찾기 정보입니다:
                        - 병원 주소: {target_address}
                        - 총 거리: {distance}m
                        - 예상 소요 시간: {duration_minutes}분
                        
                        이 정보를 자연스러운 대화체 한국어로 설명해주세요.
                        """
                        
                        natural_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "당신은 친근한 말투로 길 안내를 해주는 도우미입니다."},
                                {"role": "user", "content": directions_prompt}
                            ]
                        )
                        
                        response_text = natural_response.choices[0].message.content
                        
                        return jsonify({
                            'type': 'directions',
                            'response': response_text,
                            'route': route_result['route'],
                            'summary': {
                                'distance': distance,
                                'duration_minutes': duration_minutes
                            },
                            'start': start_coords,
                            'end': {
                                'name': target_name,
                                'address': target_address,
                                'x': target_coords['x'],
                                'y': target_coords['y']
                            }
                        })
                except Exception as e:
                    logger.error(f"길찾기 중 오류 발생: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({
                        'type': 'chat',
                        'response': '죄송합니다. 길찾기 중 오류가 발생했습니다.'
                    })
            
            elif analysis['query_type'] == 'facility_search':
                try:
                    logger.info("\n[RAG 검색 시작]")
                    logger.info(f"- 시설 유형: {analysis['place_type']}")
                    
                    # RAG 시스템으로 시설 검색
                    search_results = rag.search_similar_documents(query, analysis['place_type'], top_k=3)
                    
                    if search_results:
                        logger.info(f"검색 결과 {len(search_results)}개 찾음")
                        logger.debug(f"검색 결과 상세: {json.dumps(search_results, ensure_ascii=False, indent=2)}")
                        
                        # GPT로 자연어 응답 생성
                        response_prompt = f"""
                        F&F 신사옥 주변의 {analysis['place_type']} 검색 결과입니다:
                        {json.dumps(search_results, ensure_ascii=False, indent=2)}
                        
                        이 정보를 자연스러운 한국어로 설명해주세요. 각 시설의 이름, 위치(지하철역 기준), 영업시간, 특징적인 정보를 포함해주세요.
                        응답은 친근하고 대화체로 작성해주세요.
                        """
                        
                        natural_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "당신은 친근한 말투로 주변 시설을 추천해주는 도우미입니다."},
                                {"role": "user", "content": response_prompt}
                            ]
                        )
                        
                        response_text = natural_response.choices[0].message.content
                        logger.info(f"GPT 응답: {response_text}")
                        
                        return jsonify({
                            'type': 'chat',
                            'response': response_text
                        })
                    else:
                        logger.warning("검색 결과가 없습니다.")
                        return jsonify({
                            'type': 'chat',
                            'response': f'죄송합니다. 주변에서 {analysis["place_type"]}을(를) 찾을 수 없습니다.'
                        })
                
                except Exception as e:
                    logger.error(f"시설 검색 중 오류 발생: {str(e)}")
                    logger.error(traceback.format_exc())
                return jsonify({
                        'error': '시설 검색 중 오류가 발생했습니다.',
                        'type': 'error'
                    }), 500
            
            # 위치 검색인 경우
            elif analysis.get('is_location_query'):
                location_results = geocode_location(analysis['location_query'])
                
                if not location_results.get('addresses') or len(location_results['addresses']) == 0:
                    logger.debug(f"위치 검색 실패, 원본 쿼리로 재시도: {query}")
                    location_results = geocode_location(query)
                
                if location_results.get('addresses') and len(location_results['addresses']) > 0:
                    enhanced_results = enhance_results_with_context(
                        location_results, 
                        analysis,
                        query
                    )
                    return jsonify(enhanced_results)
                else:
                    return jsonify({
                        "status": "ZERO_RESULTS",
                        "original_query": query,
                        "addresses": []
                    })
            else:
                location_results = geocode_location(query)
                return jsonify(location_results)
                
        except Exception as e:
            logger.error(f"검색 처리 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"검색 처리 중 오류가 발생했습니다: {str(e)}"}), 500

    def analyze_query(query: str, user_location=None) -> dict:
        """
        사용자 쿼리를 분석하여 검색 유형을 결정
        Returns: 분석 결과 딕셔너리
        """
        try:
            # 길찾기 관련 키워드
            directions_keywords = ['어떻게 가', '가는 길', '가는 방법', '찾아가', '가는길', '까지 가는']
            
            # 시설 검색 관련 키워드
            search_keywords = ['어디', '있', '알려줘', '찾아줘', '검색']
            
            # 길찾기 요청인지 먼저 확인
            for keyword in directions_keywords:
                if keyword in query:
                    # 목적지 추출 (길찾기 키워드 이전의 텍스트)
                    target_location = query
                    
                    # 길찾기 키워드와 그 뒤의 텍스트 제거
                    for k in directions_keywords:
                        if k in target_location:
                            target_location = target_location.split(k)[0].strip()
                    
                    # "알려줘", "가르쳐줘" 등의 추가 텍스트 제거
                    additional_keywords = ['알려줘', '가르쳐줘', '좀 알려', '좀 가르쳐']
                    for k in additional_keywords:
                        if k in target_location:
                            target_location = target_location.split(k)[0].strip()
                    
                    logger.info(f"길찾기 요청 감지 - 정제된 목적지: {target_location}")
                    
                    return {
                        'query_type': 'directions',
                        'target_location': target_location,
                        'start_location': user_location or 'F&F 신사옥'
                    }
            
            # 시설 유형 분석 (길찾기가 아닌 경우)
            place_type = analyze_place_type(query)
            logger.info(f"분석된 시설 유형: {place_type}")
            
            # 시설 검색 요청 처리
            if any(keyword in query for keyword in search_keywords) and place_type != '없음':
                return {
                    'query_type': 'facility_search',
                    'place_type': place_type,
                    'distance_constraint': '가까운' if '가까운' in query else ''
                }
            
            # 기본값
            return {
                'query_type': 'unknown',
                'place_type': place_type
            }
            
        except Exception as e:
            logger.error(f"쿼리 분석 중 오류 발생: {str(e)}")
            return {
                'query_type': 'error',
                'error_message': str(e)
            }

    def geocode_location(location_query):
        """
        네이버 Geocoding API를 사용하여 위치 정보를 좌표로 변환합니다.
        """
        url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
            "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
        }
        params = {"query": location_query}
        
        try:
            logger.debug(f"Geocoding API 요청: {url}, 파라미터: {params}")
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            results = response.json()
            logger.debug(f"Geocoding API 응답: {results}")
            
            return results
        except Exception as e:
            logger.error(f"Geocoding API 요청 중 오류 발생: {str(e)}")
            return {"error": f"위치 검색 중 오류가 발생했습니다: {str(e)}"}

    def enhance_results_with_context(location_results, analysis, original_query):
        """
        검색 결과에 추가 컨텍스트를 더하여 강화합니다.
        """
        try:
            enhanced_results = location_results.copy()
            
            # 원본 쿼리와 분석 정보 추가
            enhanced_results["original_query"] = original_query
            enhanced_results["analysis"] = analysis
            
            # 장소 유형이 있는 경우, GPT를 활용해 결과에 추가 정보 제공
            if analysis.get('place_type'):
                # 첫 번째 주소 정보 가져오기
                first_location = location_results['addresses'][0] if location_results.get('addresses') else None
                
                if first_location:
                    place_type = analysis['place_type']
                    location_name = first_location.get('roadAddress') or first_location.get('jibunAddress')
                    
                    # GPT를 사용하여 해당 위치와 장소 유형에 대한 추가 정보 생성
                    context_prompt = f"{location_name}의 {place_type}에 대한 간략한 정보를 제공해주세요."
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"""
                            당신은 위치 기반 정보 제공 전문가입니다. 
                            특정 위치와 장소 유형에 관해 사용자가 알면 도움이 될 간략한 정보를 제공하세요.
                            답변은 3-5줄 내외로 짧고 유용하게 작성하세요.
                            """},
                            {"role": "user", "content": context_prompt}
                        ]
                    )
                    
                    additional_info = response.choices[0].message.content
                    
                    # 검색 결과에 추가 정보 포함
                    enhanced_results["additional_info"] = additional_info
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"결과 강화 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 결과 반환
            return location_results

    # 상태 확인 엔드포인트
    @app.route('/health', methods=['GET'])
    def health_check():
        logger.debug("상태 확인 요청 수신")
        return jsonify({"status": "ok"}), 200

    # 루트 경로 처리
    @app.route('/', methods=['GET'])
    def root():
        logger.debug("루트 경로 요청 수신")
        return "네이버 지도 검색 API 서버가 실행 중입니다. /api/search 엔드포인트를 통해 요청하세요."

    # favicon 요청 처리
    @app.route('/favicon.ico')
    def favicon():
        return "", 204

    # 길찾기 API 엔드포인트 추가
    @app.route('/api/directions', methods=['GET'])
    def get_directions():
        start = request.args.get('start')  # 출발지 좌표 (경도,위도)
        goal = request.args.get('goal')    # 도착지 좌표 (경도,위도)
        waypoints = request.args.get('waypoints', '')  # 경유지 (선택사항)
        option = request.args.get('option', 'trafast')  # 옵션: trafast(최속), tracomfort(편안)
        
        if not start or not goal:
            return jsonify({"error": "출발지와 도착지 좌표가 필요합니다"}), 400
        
        try:
            url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
            headers = {
                "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
                "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
            }
            params = {
                "start": start,
                "goal": goal,
                "option": option
            }
            
            if waypoints:
                params["waypoints"] = waypoints
                
            response = requests.get(url, params=params, headers=headers)
            result = response.json()
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # 주소를 좌표로 변환하는 엔드포인트 (기존 geocode 기능 활용)
    @app.route('/api/geocode', methods=['GET'])
    def geocode_address():
        address = request.args.get('address')
        
        if not address:
            return jsonify({"error": "주소가 필요합니다"}), 400
        
        try:
            # geocoding API 직접 호출
            location_data = geocode_location(address)
            if location_data.get('addresses') and len(location_data['addresses']) > 0:
                first_address = location_data['addresses'][0]
                coordinates = [float(first_address['x']), float(first_address['y'])]
                return jsonify({"coordinates": coordinates})
            else:
                return jsonify({"error": "주소를 찾을 수 없습니다"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # 길찾기 데이터를 가져오는 헬퍼 함수
    def get_directions_data(start, goal, option='trafast'):
        """
        네이버 Direction API를 사용하여 길찾기 정보를 가져옵니다.
        """
        url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
            "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
        }
        params = {
            "start": start,
            "goal": goal,
            "option": option
        }
        
        try:
            logger.debug(f"Direction API 요청: {url}, 파라미터: {params}")
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            results = response.json()
            
            # 네이버 API 응답 오류 처리
            if results.get('code') != 0:
                error_msg = results.get('message', '알 수 없는 오류')
                logger.error(f"Direction API 오류: {error_msg}")
                return {
                    "error": f"길찾기 오류: {error_msg}",
                    "code": results.get('code')
                }
            
            logger.debug(f"Direction API 응답: {results}")
            return results
        except Exception as e:
            logger.error(f"Direction API 요청 중 오류 발생: {str(e)}")
            return {"error": f"길찾기 중 오류가 발생했습니다: {str(e)}"}

    # search_similar_documents 호출 전 디버그 로깅 추가
    def debug_search(query, facility_type, top_k=3):
        logger.debug(f"""
        RAG 검색 시도:
        - 쿼리: {query}
        - 시설 유형: {facility_type}
        - top_k: {top_k}
        """)
        try:
            results = rag.search_similar_documents(query, facility_type, top_k)
            logger.debug(f"RAG 검색 결과: {json.dumps(results, ensure_ascii=False, indent=2)}")
            return results
        except Exception as e:
            logger.error(f"RAG 검색 중 오류 발생: {str(e)}\n{traceback.format_exc()}")
            return None

    def analyze_facility_type(query: str) -> tuple:
        """
        사용자 쿼리에서 시설 유형을 분석합니다.
        Returns: (facility_type, confidence_score, matched_keyword)
        """
        facility_keywords = {
            '병원': {
                'high_confidence': ['병원', '의원', '치과', '한의원'],
                'medium_confidence': ['진료', '진찰', '의료', '검진'],
                'context': ['아파서', '아픈데', '아프다', '치료', '진단', '가까운']
            },
            '음식점': {
                'high_confidence': ['음식점', '식당', '레스토랑', '맛집'],
                'medium_confidence': ['밥집', '식사'],
                'context': ['배고파', '먹을', '먹고', '식사', '가까운']
            },
            '카페': {
                'high_confidence': ['카페', '커피숍', '커피집'],
                'medium_confidence': ['커피', '디저트', '베이커리'],
                'context': ['마시고', '달달한', '차', '가까운']
            },
            '약국': {
                'high_confidence': ['약국', '약방', '드럭스토어'],
                'medium_confidence': ['약', '처방', '조제'],
                'context': ['약이', '약을', '처방전', '가까운']
            }
        }
        
        logger.info("\n=== 시설 유형 분석 시작 ===")
        logger.info(f"분석할 쿼리: '{query}'")
        
        best_match = None
        highest_score = 0
        matched_keyword = None
        
        for facility_type, keywords in facility_keywords.items():
            current_score = 0
            current_keyword = None
            
            # 높은 신뢰도 키워드 체크 (가중치: 1.0)
            for keyword in keywords['high_confidence']:
                if keyword in query:
                    current_score = 1.0
                    current_keyword = keyword
                    logger.info(f"- 높은 신뢰도 키워드 발견: '{keyword}' in '{facility_type}'")
                    break
            
            # 중간 신뢰도 키워드 체크 (가중치: 0.7)
            if current_score == 0:
                for keyword in keywords['medium_confidence']:
                    if keyword in query:
                        current_score = 0.7
                        current_keyword = keyword
                        logger.info(f"- 중간 신뢰도 키워드 발견: '{keyword}' in '{facility_type}'")
                        break
            
            # 문맥 키워드 체크 (가중치: 0.3)
            if current_score == 0:
                for keyword in keywords['context']:
                    if keyword in query:
                        current_score = 0.3
                        current_keyword = keyword
                        logger.info(f"- 문맥 키워드 발견: '{keyword}' in '{facility_type}'")
                        break
            
            if current_score > highest_score:
                highest_score = current_score
                best_match = facility_type
                matched_keyword = current_keyword
        
        logger.info(f"=== 분석 결과 ===")
        logger.info(f"- 시설 유형: {best_match or '없음'}")
        logger.info(f"- 신뢰도 점수: {highest_score}")
        logger.info(f"- 매칭된 키워드: {matched_keyword or '없음'}")
        
        return best_match, highest_score, matched_keyword

    def analyze_place_type(query: str) -> str:
        """
        사용자 쿼리에서 시설 유형을 분석합니다.
        Returns: '병원', '음식점', '카페', '약국', '없음' 중 하나
        """
        facility_keywords = {
            '병원': {
                'high_confidence': ['병원', '의원', '치과', '한의원'],
                'medium_confidence': ['진료', '진찰', '의료', '검진'],
                'context': ['아파서', '아픈데', '아프다', '치료', '진단', '가까운']
            },
            '음식점': {
                'high_confidence': ['음식점', '식당', '레스토랑', '맛집'],
                'medium_confidence': ['밥집', '식사'],
                'context': ['배고파', '먹을', '먹고', '식사', '가까운']
            },
            '카페': {
                'high_confidence': ['카페', '커피숍', '커피집'],
                'medium_confidence': ['커피', '디저트', '베이커리'],
                'context': ['마시고', '달달한', '차', '가까운']
            },
            '약국': {
                'high_confidence': ['약국', '약방', '드럭스토어'],
                'medium_confidence': ['약', '처방', '조제'],
                'context': ['약이', '약을', '처방전', '가까운']
            }
        }
        
        logger.info("\n=== 시설 유형 분석 시작 ===")
        logger.info(f"분석할 쿼리: '{query}'")
        
        best_match = None
        highest_score = 0
        
        for facility_type, keywords in facility_keywords.items():
            current_score = 0
            
            # 높은 신뢰도 키워드 체크 (가중치: 1.0)
            for keyword in keywords['high_confidence']:
                if keyword in query:
                    current_score = 1.0
                    logger.info(f"- 높은 신뢰도 키워드 발견: '{keyword}' in '{facility_type}'")
                    break
            
            # 중간 신뢰도 키워드 체크 (가중치: 0.7)
            if current_score == 0:
                for keyword in keywords['medium_confidence']:
                    if keyword in query:
                        current_score = 0.7
                        logger.info(f"- 중간 신뢰도 키워드 발견: '{keyword}' in '{facility_type}'")
                        break
            
            # 문맥 키워드 체크 (가중치: 0.3)
            if current_score == 0:
                for keyword in keywords['context']:
                    if keyword in query:
                        current_score = 0.3
                        logger.info(f"- 문맥 키워드 발견: '{keyword}' in '{facility_type}'")
                        break
            
            if current_score > highest_score:
                highest_score = current_score
                best_match = facility_type
        
        logger.info(f"=== 분석 결과 ===")
        logger.info(f"- 시설 유형: {best_match or '없음'}")
        logger.info(f"- 신뢰도 점수: {highest_score}")
        
        return best_match or '없음'

    if __name__ == '__main__':
        try:
            logger.info("백엔드 서버 시작: http://localhost:5000")
            app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
        except Exception as e:
            logger.error(f"서버 시작 중 오류 발생: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
except Exception as e:
    logger.error(f"백엔드 초기화 중 오류 발생: {str(e)}")
    traceback.print_exc()
    sys.exit(1) 