import React from 'react';
import { Document, Page, Text, View, StyleSheet, Font } from '@react-pdf/renderer';
import NotoSansKR from '../fonts/NotoSansKR-Regular.ttf';

// 폰트 등록
Font.register({
  family: 'NotoSansKR',
  src: NotoSansKR,
});

// 날짜 및 시간 포맷 함수
const formatDate = (isoString) => {
  if (!isoString) return '날짜 없음';
  const date = new Date(isoString);
  return date.toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
};

// 시간 포맷 함수
const formatTimeOnly = (isoString) => {
  if (!isoString) return '시간 없음';
  const date = new Date(isoString);
  return date.toLocaleTimeString('ko-KR', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

// 적재물 양식에 사용하는 날짜+시간 포맷함수
const formatDateTime = (isoString) => {
  if (!isoString) return '날짜 없음';
  const date = new Date(isoString);
  return date.toLocaleString('ko-KR', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

// 스타일 정의
const styles = StyleSheet.create({
  page: {
    padding: 30,
    fontFamily: 'NotoSansKR',
  },
  header: {
    fontSize: 16,
    marginBottom: 20,
    textAlign: 'center',
  },
  table: {
    display: 'table',
    width: 'auto',
    borderStyle: 'solid',
    borderWidth: 1,
    borderColor: '#000',
  },
  tableRow: {
    flexDirection: 'row',
  },
  tableCol: {
    borderStyle: 'solid',
    borderWidth: 1,
    borderColor: '#000',
  },
  tableCell: {
    margin: 2,
    fontSize: 10,
    padding: 4,
  },
  idCol: { width: '10%' },
  descriptionCol: { width: '30%' },
  detectedTimeCol: { width: '60%' },
  violationCol: { width: '25%' }, // 위반 차량 열의 비율
  dateCol: { width: '25%' }, // 위반 날짜 열의 비율
  timeCol: { width: '25%' }, // 위반 시간 열의 비율
  locationCol: { width: '25%' }, // 위반 장소 열의 비율
});

const PDF_form = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <Document>
        <Page size="A4" style={styles.page}>
          <Text style={styles.header}>데이터 내역</Text>
          <Text>표시할 데이터가 없습니다.</Text>
        </Page>
      </Document>
    );
  }

  // 데이터 구조에 따라 양식 선택
  const isViolationData = data.some(item => item.hasOwnProperty('violation_number')); // 위반 차량 데이터가 있는지 확인

  return (
    <Document>
      <Page size="A4" style={styles.page}>
        <Text style={styles.header}>{isViolationData ? '위반 차량 내역' : '적재물 내역'}</Text>
        <View style={styles.table}>
          {/* 테이블 헤더 */}
          <View style={styles.tableRow}>
            {isViolationData ? (
              <>
                <View style={[styles.tableCol, styles.violationCol]}>
                  <Text style={styles.tableCell}>위반차량</Text>
                </View>
                <View style={[styles.tableCol, styles.dateCol]}>
                  <Text style={styles.tableCell}>위반날짜</Text>
                </View>
                <View style={[styles.tableCol, styles.timeCol]}>
                  <Text style={styles.tableCell}>위반시간</Text>
                </View>
                <View style={[styles.tableCol, styles.locationCol]}>
                  <Text style={styles.tableCell}>위반장소</Text>
                </View>
              </>
            ) : (
              <>
                <View style={[styles.tableCol, styles.idCol]}>
                  <Text style={styles.tableCell}>ID</Text>
                </View>
                <View style={[styles.tableCol, styles.descriptionCol]}>
                  <Text style={styles.tableCell}>설명</Text>
                </View>
                <View style={[styles.tableCol, styles.detectedTimeCol]}>
                  <Text style={styles.tableCell}>탐지 시간</Text>
                </View>
              </>
            )}
          </View>
          
          {/* 테이블 데이터 */}
          {data.map((item, index) => (
            <View key={index} style={styles.tableRow}>
              {isViolationData ? (
                // 위반 차량 데이터 렌더링
                <>
                  <View style={[styles.tableCol, styles.violationCol]}>
                    <Text style={styles.tableCell}>{item.violation_number}</Text>
                  </View>
                  <View style={[styles.tableCol, styles.dateCol]}>
                    <Text style={styles.tableCell}>{formatDate(item.upload_time)}</Text>
                  </View>
                  <View style={[styles.tableCol, styles.timeCol]}>
                    <Text style={styles.tableCell}>{formatTimeOnly(item.upload_time)}</Text>
                  </View>
                  <View style={[styles.tableCol, styles.locationCol]}>
                    <Text style={styles.tableCell}>{item.violation_location || '정보 없음'}</Text>
                  </View>
                </>
              ) : (
                // 적재물 데이터 렌더링
                <>
                  <View style={[styles.tableCol, styles.idCol]}>
                    <Text style={styles.tableCell}>{item.id}</Text>
                  </View>
                  <View style={[styles.tableCol, styles.descriptionCol]}>
                    <Text style={styles.tableCell}>{item.description}</Text>
                  </View>
                  <View style={[styles.tableCol, styles.detectedTimeCol]}>
                    <Text style={styles.tableCell}>{formatDateTime(item.detected_at)}</Text>
                  </View>
                </>
              )}
            </View>
          ))}
        </View>
      </Page>
    </Document>
  );
};

export default PDF_form;
