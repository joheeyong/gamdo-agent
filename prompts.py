"""Claude에게 전달할 시스템/유저 프롬프트 모음."""

ANALYZE_USER_SYSTEM = """\
당신은 소셜 미디어 사진 스타일 분석 전문가입니다.
사용자의 게시글, 피드, 스토리 데이터를 분석하여 사진 취향과 톤앤매너를 파악합니다.
반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 텍스트는 포함하지 마세요.
"""

ANALYZE_USER_PROMPT = """\
아래 사용자의 소셜 미디어 데이터를 분석하고, JSON 형식으로 사용자의 사진 스타일 프로필을 만들어 주세요.

=== 게시글 ===
{posts}

=== 피드 ===
{feeds}

=== 스토리 ===
{stories}

다음 JSON 형식으로 응답해 주세요:
{{
  "styleProfile": {{
    "primaryStyle": "미니멀 | 빈티지 | 모던 | 내추럴 | 드라마틱 | 파스텔 | 다크 | 필름",
    "secondaryStyles": ["보조 스타일1", "보조 스타일2"],
    "colorPreference": {{
      "preferredTones": "warm | cool | neutral | mixed",
      "dominantColors": ["#HEX1", "#HEX2", "#HEX3"],
      "saturationTendency": "high | medium | low",
      "brightnessTendency": "high | medium | low"
    }},
    "compositionPreference": {{
      "preferredTechniques": ["삼분법", "중앙배치"],
      "subjectPreference": "인물 | 풍경 | 음식 | 사물 | 혼합",
      "croppingStyle": "tight | balanced | wide"
    }},
    "moodKeywords": ["키워드1", "키워드2", "키워드3"],
    "editingStyle": {{
      "filterTendency": "heavy | moderate | minimal | none",
      "contrastLevel": "high | medium | low",
      "description": "한국어로 이 사용자의 보정 성향 설명 (2-3문장)"
    }}
  }},
  "summary": "한국어로 이 사용자의 전체적인 사진 스타일을 설명하는 요약 (3-5문장)",
  "recommendations": [
    "이 사용자에게 추천하는 촬영/보정 방향 1",
    "이 사용자에게 추천하는 촬영/보정 방향 2",
    "이 사용자에게 추천하는 촬영/보정 방향 3"
  ]
}}
"""

TRANSFORM_PHOTO_SYSTEM = """\
당신은 전문 사진 보정 코칭 전문가입니다.
사용자의 스타일 프로필에 맞춰 업로드된 사진에 대한 맞춤형 보정 가이드를 제공합니다.
반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 텍스트는 포함하지 마세요.
"""

TRANSFORM_PHOTO_PROMPT = """\
아래 사용자의 스타일 프로필을 참고하여, 첨부된 사진을 이 사용자의 스타일에 맞게 변형하기 위한 보정 가이드를 제공해 주세요.

=== 사용자 스타일 프로필 ===
{style_profile}

다음 JSON 형식으로 응답해 주세요:
{{
  "currentAnalysis": {{
    "dominantColors": ["#HEX1", "#HEX2", "#HEX3", "#HEX4", "#HEX5"],
    "colorTemperature": "warm | cool | neutral",
    "saturationLevel": 0.0,
    "brightnessLevel": 0.0,
    "compositionTechnique": "삼분법 | 중앙배치 | 대각선 등",
    "overallMood": "현재 사진의 분위기"
  }},
  "transformGuide": {{
    "targetStyle": "변환 목표 스타일",
    "colorAdjustments": {{
      "temperature": {{ "direction": "warmer | cooler | keep", "amount": "slight | moderate | strong" }},
      "saturation": {{ "direction": "increase | decrease | keep", "amount": "slight | moderate | strong" }},
      "brightness": {{ "direction": "increase | decrease | keep", "amount": "slight | moderate | strong" }},
      "contrast": {{ "direction": "increase | decrease | keep", "amount": "slight | moderate | strong" }},
      "targetPalette": ["#HEX1", "#HEX2", "#HEX3"]
    }},
    "compositionSuggestions": [
      "구도 관련 제안 1",
      "구도 관련 제안 2"
    ],
    "editingSteps": [
      {{ "step": 1, "tool": "밝기/노출", "action": "구체적인 보정 지시 (한국어)" }},
      {{ "step": 2, "tool": "색온도", "action": "구체적인 보정 지시 (한국어)" }},
      {{ "step": 3, "tool": "채도", "action": "구체적인 보정 지시 (한국어)" }},
      {{ "step": 4, "tool": "대비", "action": "구체적인 보정 지시 (한국어)" }},
      {{ "step": 5, "tool": "필터/프리셋", "action": "추천 필터 또는 프리셋 (한국어)" }}
    ],
    "appRecommendations": [
      {{ "app": "앱 이름", "reason": "추천 이유 (한국어)" }}
    ]
  }},
  "beforeAfterDescription": "변환 전후 차이를 설명하는 한국어 문장 (3-5문장)",
  "overallScore": {{
    "current": 0,
    "expectedAfter": 0
  }}
}}
"""
