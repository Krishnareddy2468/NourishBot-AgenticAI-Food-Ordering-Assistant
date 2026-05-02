import { useState, useEffect } from 'react'
import * as XLSX from 'xlsx'

// Emoji guesser based on item name + category
function getEmoji(name = '', category = '') {
  const n = name.toLowerCase()
  if (n.includes('biryani')) return '🍚'
  if (n.includes('butter chicken') || n.includes('butter chic')) return '🍲'
  if (n.includes('chicken')) return '🍗'
  if (n.includes('mutton') || n.includes('rogan')) return '🥩'
  if (n.includes('fish')) return '🐟'
  if (n.includes('prawn') || n.includes('shrimp')) return '🦐'
  if (n.includes('paneer tikka')) return '🔥'
  if (n.includes('paneer')) return '🧀'
  if (n.includes('tikka') || n.includes('kebab') || n.includes('kebab')) return '🍢'
  if (n.includes('dosa')) return '🥞'
  if (n.includes('naan') || n.includes('roti') || n.includes('bread') || n.includes('bhature')) return '🫓'
  if (n.includes('fried rice')) return '🍳'
  if (n.includes('rice')) return '🍚'
  if (n.includes('noodles')) return '🍜'
  if (n.includes('manchurian')) return '🫙'
  if (n.includes('aloo') || n.includes('gobi')) return '🥔'
  if (n.includes('palak') || n.includes('spinach')) return '🥬'
  if (n.includes('mushroom')) return '🍄'
  if (n.includes('chole') || n.includes('chana')) return '🫘'
  if (n.includes('dal') || n.includes('curry')) return '🍛'
  if (n.includes('thali')) return '🍱'
  if (n.includes('family')) return '👨‍👩‍👧‍👦'
  if (n.includes('combo') || n.includes('pack')) return '🎉'
  if (n.includes('lassi') || n.includes('mango')) return '🥭'
  if (n.includes('chai') || n.includes('tea') || n.includes('coffee')) return '☕'
  if (n.includes('lime') || n.includes('lemon')) return '🍋'
  if (n.includes('juice') || n.includes('soda')) return '🥤'
  if (n.includes('gulab') || n.includes('jamun')) return '🔮'
  if (n.includes('rasmalai') || n.includes('kheer') || n.includes('pudding')) return '🍮'
  if (n.includes('ice cream') || n.includes('kulfi')) return '🍨'
  if (category === 'Non-Veg') return '🍽️'
  if (category === 'Beverages') return '🥤'
  if (category === 'Desserts') return '🍨'
  if (category === 'Combos') return '🍱'
  return '🥗'
}

// Safe string converter
const s = (val, fallback = '') => (val !== null && val !== undefined ? String(val).trim() : fallback)
const n = (val, fallback = 0) => { const num = Number(val); return isFinite(num) ? num : fallback }

// Parse Excel date serial / Date / string to display string
function parseDate(val) {
  if (!val) return ''
  if (val instanceof Date) return val.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: '2-digit' })
  if (typeof val === 'string') {
    if (val.length > 20 || val.includes('GMT')) {
      const d = new Date(val);
      if (!isNaN(d.getTime())) {
        return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: '2-digit' }) + ' ' + d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })
      }
    }
    return val
  }
  return String(val)
}

export function useExcelData() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastLoaded, setLastLoaded] = useState(null)

  async function loadExcel() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/Project_Dzukku.xlsx?t=' + Date.now())
      if (!res.ok) throw new Error(`Failed to fetch Excel: ${res.status}`)
      const arrayBuffer = await res.arrayBuffer()
      const wb = XLSX.read(arrayBuffer, { type: 'array', cellDates: true })

      const result = {}

      // ─── 1. MENU (from "menu card" sheet) ────────────────────────────
      const menuSheetName = wb.SheetNames.find(name =>
        name.toLowerCase() === 'menu card' || name.toLowerCase() === 'master_menu'
      ) || wb.SheetNames[0]

      const menuRows = XLSX.utils.sheet_to_json(wb.Sheets[menuSheetName], {
        header: 1, defval: null, blankrows: false,
      })

      // Row 0 = headers, so skip it
      result.menu = menuRows
        .slice(1)
        .filter(row => row[0] && row[2]) // must have ID + Item Name
        .map(row => ({
          id: s(row[0]),
          category: s(row[1], 'Veg'),
          name: s(row[2]),
          description: s(row[3]),
          price: n(row[4]),
          status: s(row[5], 'Available'),
          isSpecial: s(row[6]).toLowerCase() === 'yes',
          stock: n(row[7]),
          specialPrice: row[8] !== null ? n(row[8]) : null,
          prepTime: s(row[9], '15 mins'),
          recipe: s(row[10]),
          emoji: getEmoji(s(row[2]), s(row[1])),
        }))

      // ─── 2. ORDERS (from "Orders" sheet) ─────────────────────────────
      const ordersRows = XLSX.utils.sheet_to_json(wb.Sheets['Orders'] || {}, { defval: '' })
      result.orders = ordersRows
        .filter(row => row['OrderID'])
        .map(row => ({
          id: s(row['OrderID']),
          customer: s(row['customer'], 'Guest'),
          phone: s(row['Phone'], '-'),
          item: s(row['Item']),
          price: n(row['Total']),
          status: s(row['Status'], 'Pending'),
          dateTime: parseDate(row['Date/Time']),
          deliveryDate: parseDate(row['DeliveryDate']),
          platform: s(row['Platform'], 'Offline'),
          address: s(row['Address']),
          qty: n(row['Qty'], 1),
          unitPrice: n(row['UnitPrice']),
          invoiceUrl: s(row['InvoiceURL']),
          special: s(row['special ']),
          // for KDS/queue compatibility
          eta: '20 mins',
          items: [{ name: s(row['Item']), qty: n(row['Qty'], 1), emoji: '🍽️' }],
        }))

      // ─── 3. ANALYTICS (from "Order Analytical" sheet) ─────────────────
      const analyticsRows = XLSX.utils.sheet_to_json(wb.Sheets['Order Analytical'] || {}, { defval: 0 })
      result.analytics = analyticsRows
        .filter(row => row['Date'])
        .map(row => ({
          date: parseDate(row['Date']),
          orders: n(row['Total_Orders']),
          revenue: n(row['Total_Revenue']),
          online: n(row['Online_Payments']),
          cash: n(row['Cash_Payments']),
          avg: n(row['Avg_Order_Value']),
          delivery: n(row['Delivery_Orders']),
        }))
        .reverse() // oldest first for charts

      // ─── 4. SPECIAL ITEMS (from "Special items" sheet) ────────────────
      const specialRows = XLSX.utils.sheet_to_json(wb.Sheets['Special items'] || {}, { defval: '' })
      result.specials = specialRows
        .filter(row => row['Special_ID'])
        .map(row => ({
          id: s(row['Special_ID']),
          itemId: s(row['Item_ID']),
          name: s(row['Item_Name']),
          category: s(row['Category']),
          normalPrice: n(row['Normal_Price']),
          // stored as 0.14 = 14%, convert to integer %
          discount: Math.round(n(row['Discount_Percent']) * 100),
          emoji: getEmoji(s(row['Item_Name'])),
        }))

      // ─── 5. TABLES (from "Tables" sheet) ──────────────────────────────
      const tablesRows = XLSX.utils.sheet_to_json(wb.Sheets['Tables'] || {}, { defval: '' })
      result.tables = tablesRows
        .filter(row => row['Tables'])
        .map((row, i) => ({
          id: s(row['Tables']),
          status: s(row['status'], 'Available'),
          capacity: i < 4 ? 2 : i < 14 ? 4 : 6,
          section: i < 7 ? 'Ground Floor' : i < 14 ? 'First Floor' : 'Terrace',
        }))

      // ─── 6. RESERVATIONS (from "Reservation" sheet) ───────────────────
      const resWs = wb.Sheets['Reservation']
      if (resWs) {
        const resRows = XLSX.utils.sheet_to_json(resWs, { defval: '' })
        result.reservations = resRows
          .filter(row => row['Res_ID'])
          .map(row => ({
            id: s(row['Res_ID']),
            customer: s(row['Customer_Name']),
            phone: s(row['Phone']),
            date: parseDate(row['Date']),
            time: row['Time'] instanceof Date
              ? row['Time'].toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })
              : s(row['Time']),
            guests: n(row['Guests'], 2),
            table: s(row['Table_No']),
            status: s(row['Status'], 'Confirmed'),
            requests: s(row['Special_Requests']),
            email: s(row['Email']),
          }))
      } else {
        result.reservations = []
      }

      // ─── 7. EMPLOYEES (from "Employees" sheet) ────────────────────────
      const empRows = XLSX.utils.sheet_to_json(wb.Sheets['Employees'] || {}, { defval: '' })
      result.employees = empRows
        .filter(row => row['Employee_ID'])
        .map(row => ({
          id: s(row['Employee_ID']),
          name: s(row['Name']),
          role: s(row['Role']),
          dept: s(row['Department']),
          phone: s(row['Phone']),
          email: s(row['Email']),
          shift: s(row['Shift']),
          salary: n(row['Salary']),
          status: s(row['Status'], 'Active'),
          rating: n(row['Performance_Rating']),
          skills: s(row['Skills']),
          joined: parseDate(row['Date_of_Joining']),
        }))

      // ─── 8. OFFLINE ORDERS (from "Dashboard_Offline" sheet) ───────────
      const offlineWs = wb.Sheets['Dashboard_Offline']
      if (offlineWs) {
        const offlineRows = XLSX.utils.sheet_to_json(offlineWs, { defval: '' })
        result.offlineOrders = offlineRows
          .filter(row => row['OrderID'])
          .map(row => ({
            id: s(row['OrderID']),
            customer: s(row['Customer Name'] || row['customer']),
            phone: s(row['Phone']),
            platform: s(row['Platform'], 'Offline'),
            dateTime: parseDate(row['Date/Time']),
            status: s(row['Status'], 'Pending'),
            eta: '20 mins',
            item: 'Offline Order',
            price: 0,
            items: [],
          }))
      } else {
        result.offlineOrders = []
      }

      // ─── 9. SALES DASHBOARD (from "Sales Dashboard" sheet) ────────────
      const salesWs = wb.Sheets['Sales Dashboard']
      if (salesWs) {
        const salesRows = XLSX.utils.sheet_to_json(salesWs, { header: 1, defval: null })
        // Col A=Date, Col B=TotalSales, Col D=Date, Col E=Item, Col F=Qty
        result.salesByItem = salesRows
          .slice(1)
          .filter(row => row[4] && row[5]) // Item + Qty
          .map(row => ({
            date: parseDate(row[3]),
            item: s(row[4]),
            qty: n(row[5]),
          }))

        result.salesByDay = salesRows
          .slice(1)
          .filter(row => row[0] instanceof Date && row[1] !== null)
          .map(row => ({
            date: parseDate(row[0]),
            total: n(row[1]),
          }))
      } else {
        result.salesByItem = []
        result.salesByDay = []
      }

      // ─── 10. INVOICES (from "Invoices" sheet) ─────────────────────────
      const invoicesWs = wb.Sheets['Invoices']
      if (invoicesWs) {
        const invoiceRows = XLSX.utils.sheet_to_json(invoicesWs, { header: 1, defval: null })
        result.invoices = invoiceRows
          .filter(row => row[0])
          .map(row => ({
            orderId: s(row[0]),
            customer: s(row[1]),
            dateTime: parseDate(row[2]),
            url: s(row[3]),
            amount: n(row[4]),
            status: s(row[5]),
          }))
      } else {
        result.invoices = []
      }

      setData(result)
      setLastLoaded(new Date())
    } catch (err) {
      console.error('[useExcelData] Error:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadExcel()
    // Auto-poll the Excel file every 10 seconds for updates
    const intervalId = setInterval(() => {
      loadExcel()
    }, 10000)
    return () => clearInterval(intervalId)
  }, [])

  return { data, loading, error, reload: loadExcel, lastLoaded }
}
